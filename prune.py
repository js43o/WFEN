#!/usr/bin/env python3
"""CV72 ACL sparse-pruning pilot for MultiFrameWaveBFR (WACV/Breeze variants).

Run from the WaveBFR project root (next to train.py). Example:
  python prune_wacv6m_cv72.py --name wacv_ver_mf --mode inspect
  python prune_wacv6m_cv72.py --name wacv_ver_mf --mode prune --device cuda

Defaults:
  source checkpoint: check_points/<name>/latest_net_G.pth
  outputs:           ptp/<name>/<name>_p50.{pth,onnx,yaml}
No fine-tuning is performed. This is a CV72 sparse-acceleration feasibility test.
"""

import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path

import torch
from torch import nn

from models.arch.multi_frame_wavebfr import MultiFrameWaveBFR

SHAPE = (1, 3, 112, 1120)
PAD = 128.0 / 127.5 - 1.0
ALLOWED_MISSING_FIXED = {
    "lf_down1_2.body.1.conv.weight",
    "lf_down2_3.body.1.conv.weight",
    "hf_down1_2.body.1.conv.weight",
    "hf_down2_3.body.1.conv.weight",
}


class CV72StripWrapper(nn.Module):
    """Pre-normalized RGB strip [-1,1] -> one RGB SR image [-1,1]."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network.forward_strip(x)


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # only load checkpoints from trusted sources
    obj = torch.load(str(path), **kwargs)
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "params_ema",
            "params",
            "model",
            "generator",
            "netG",
            "net_g",
        ):
            if isinstance(obj.get(k), dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a PyTorch state_dict, got {type(obj).__name__}")
    state = {}
    for name, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while name.startswith(("module.", "model.", "network.", "netG.", "net_g.")):
            name = name.split(".", 1)[1]
        if name in state:
            raise RuntimeError(f"Duplicate key after prefix normalization: {name}")
        state[name] = value
    return state


def block_count(state, prefix):
    indexes = {
        int(m.group(1))
        for key in state
        if (m := re.match(r"^" + re.escape(prefix) + r"\.(\d+)\.", key))
    }
    if not indexes or indexes != set(range(max(indexes) + 1)):
        raise RuntimeError(
            f"Could not infer consecutive Transformer blocks at '{prefix}'"
        )
    return max(indexes) + 1


def infer_architecture(state):
    """Infer supported architecture hyperparameters from checkpoint tensor keys/shapes."""
    if "patch_embed.proj.weight" not in state:
        raise RuntimeError(
            "Not a supported MultiFrameWaveBFR checkpoint: patch_embed.proj.weight absent"
        )
    if not any(k.startswith("feature_accumulator.") for k in state):
        raise RuntimeError(
            "This looks like a single-frame checkpoint, not trained multi-frame weights"
        )

    dim = int(state["patch_embed.proj.weight"].shape[0])
    backbone = (
        "wavebfr" if any(k.startswith("lf_encoder_lv3.") for k in state) else "breeze"
    )
    stage_names = ["encoder_lv1", "encoder_lv2"]
    if backbone == "wavebfr":
        stage_names.append("encoder_lv3")
    stage_names.append("latent")

    counts, heads, ffn_interval = [], [], []
    for stage in stage_names:
        prefix = f"lf_{stage}"
        counts.append(block_count(state, prefix))
        tname = f"{prefix}.0.attn.temperature"
        if tname not in state:
            raise RuntimeError(f"Cannot infer attention heads: '{tname}' missing")
        heads.append(int(state[tname].shape[0]))
        fname = f"{prefix}.0.ffn.project_in.weight"
        if fname not in state:
            raise RuntimeError(f"Cannot infer FFN dimension: '{fname}' missing")
        out_features, in_features = state[fname].shape[:2]
        if out_features % 2:
            raise RuntimeError(f"Unexpected FFN project_in shape at '{fname}'")
        hidden = out_features // 2
        ffn_interval.append((hidden / in_features, (hidden + 1) / in_features))

    # The constructor uses int(dim * ffn_expansion_factor); many real-valued
    # factors map to the same integer widths, so prefer the training default.
    lower = max(item[0] for item in ffn_interval)
    upper = min(item[1] for item in ffn_interval)
    if not lower < upper:
        raise RuntimeError(
            "FFN expansion ratios are inconsistent across checkpoint stages"
        )
    factor = 2.66 if lower <= 2.66 < upper else (lower + upper) / 2

    if backbone == "wavebfr":
        n_refine = block_count(state, "last_refinement")
    else:
        n_refine = 2  # no refinement layer in Breeze model; unused parameter

    norm_type = (
        "WithBias" if "lf_encoder_lv1.0.norm1.body.bias" in state else "BiasFree"
    )
    bias = "lf_encoder_lv1.0.attn.qkv.bias" in state
    return {
        "backbone": backbone,
        "dim": dim,
        "num_blocks": tuple(counts),
        "heads": tuple(heads),
        "num_refinement_blocks": n_refine,
        "ffn_expansion_factor": factor,
        "bias": bias,
        "LayerNorm_type": norm_type,
    }


def load_weights(net, state, checkpoint_path):
    original_fixed = {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if not p.requires_grad
    }
    missing, unexpected = net.load_state_dict(state, strict=False)
    illegal_missing = sorted(set(missing) - ALLOWED_MISSING_FIXED)
    if illegal_missing or unexpected:
        raise RuntimeError(
            f"Incomplete checkpoint {checkpoint_path}; "
            f"missing={illegal_missing[:20]}, unexpected={list(unexpected)[:20]}"
        )
    # The fixed transforms must have their exact mathematically defined weights.
    current = dict(net.named_parameters())
    for name, baseline in original_fixed.items():
        if not torch.equal(current[name].detach().cpu(), baseline):
            raise RuntimeError(
                f"Corrupted fixed convolution / wavelet weight in checkpoint: {name}"
            )
    if missing:
        print("Checkpoint loaded. FixedPixelUnshuffle weights initialized.")
    else:
        print("Checkpoint loaded. All checkpoint tensors match.")
    print(f"[weights] {len(state)} tensors from {checkpoint_path}")


def make_network(checkpoint):
    state = read_checkpoint(checkpoint)
    config = infer_architecture(state)
    print("[architecture]", json.dumps(config, ensure_ascii=False))
    net = MultiFrameWaveBFR(**config).eval().float()
    load_weights(net, state, checkpoint)
    return net, config


def frozen_snapshot(net):
    return {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if not p.requires_grad
    }


def assert_frozen(net, before):
    current = dict(net.named_parameters())
    for n, original in before.items():
        if current[n].requires_grad:
            raise RuntimeError(f"CRITICAL: frozen parameter became trainable: {n}")
        if not torch.equal(current[n].detach().cpu(), original):
            raise RuntimeError(f"CRITICAL: frozen weight modified during pruning: {n}")
    print(f"[safety] {len(before)} frozen tensors are bitwise unchanged")


def report_sparsity(net):
    frozen = learnable = zeros_frozen = zeros_learnable = 0
    for _, p in net.named_parameters():
        if p.ndim < 2:
            continue
        total = p.numel()
        zeros = int((p.detach() == 0).sum().item())
        if p.requires_grad:
            learnable += total
            zeros_learnable += zeros
        else:
            frozen += total
            zeros_frozen += zeros
    ratio = zeros_learnable / max(1, learnable)
    print(
        f"[sparsity] learnable 2D+ weights: {zeros_learnable:,}/{learnable:,} = {ratio:.2%}"
    )
    print(f"[sparsity] frozen weights excluded: {zeros_frozen:,}/{frozen:,} zeros")
    return {
        "learnable_zeros": zeros_learnable,
        "learnable_weights": learnable,
        "learnable_sparsity": ratio,
        "frozen_weights": frozen,
        "frozen_zeros": zeros_frozen,
    }


def representative_input():
    x = torch.full(SHAPE, PAD, dtype=torch.float32)
    gen = torch.Generator().manual_seed(42)
    x[:, :, :, : 112 * 6] = torch.rand((1, 3, 112, 112 * 6), generator=gen) * 2 - 1
    return x


def verify_strip(net):
    x = representative_input()
    frames = torch.stack(x.split(112, dim=-1), dim=1)
    mask = torch.arange(10).unsqueeze(0) < 6
    net.eval()
    with torch.inference_mode():
        explicit = net(frames, frame_mask=mask)
        strip = net.forward_strip(x)
    diff = float((explicit - strip).abs().max())
    print(f"[strip parity] max abs diff={diff:.7g}")
    if not torch.allclose(explicit, strip, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "Explicit-mask and strip forward disagree; abort ONNX export"
        )


def acl_l1_prune(wrapper, target, workdir, device):
    os.environ.setdefault("ACL_LOG_DIR", str(workdir / "acl_logs"))
    Path(os.environ["ACL_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import AmbaCompressionLibrary.torch as acl
    except ImportError as e:
        raise RuntimeError(
            "AmbaCompressionLibrary not installed. Use the SpongeKit Python environment."
        ) from e
    no_prune = ["wavelet_transform", "body.1.conv", "dwconv"]
    print("[ACL] exclude keywords:", no_prune)
    try:
        model_info = acl.TorchModelInfo(
            wrapper, input_shape=SHAPE, no_prune_keywords=no_prune
        )
    except TypeError as e:
        raise RuntimeError(
            "Your ACL TorchModelInfo does not accept no_prune_keywords"
        ) from e
    algorithm = acl.algorithm.Uniform("cv72")
    criterion = acl.criterion.L1Unstructured()
    schedule = acl.schedule.OneShot(
        target_sparsity=target, sparsity_metric="coefficient"
    )
    pruner = acl.Pruner(model_info, algorithm, criterion, schedule)
    start = time.time()
    print("[ACL] analyzing 10-slot multi-frame graph; may take time on CPU", flush=True)
    pruner.analyze(verbose=0, export=False)
    masks, weights = pruner.compute_masks(0)
    pruner.apply_masks(wrapper, masks, weights)
    pruner.update_model(wrapper, update_mask=True)
    print(f"[ACL] completed in {time.time() - start:.1f}s")
    try:
        print(
            "[ACL] coefficient sparsity",
            model_info.coefficient_sparsity(),
            "MAC sparsity",
            model_info.mac_sparsity(),
        )
    except Exception as exc:
        print("[ACL] optional sparsity report unavailable:", exc)


def save_checkpoint(path, net, target, model_name, architecture, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = {
        n: (p.detach().cpu() != 0).to(torch.uint8)
        for n, p in net.named_parameters()
        if p.requires_grad and p.ndim >= 2
    }
    torch.save(
        {
            "state_dict": net.state_dict(),
            "masks": masks,
            "meta": {
                "model_name": model_name,
                "target_sparsity": target,
                "architecture": architecture,
                **extra,
            },
        },
        path,
    )
    print("[saved]", path)


def export_onnx(net, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.eval().float().cpu()
    wrapper = CV72StripWrapper(net)
    verify_strip(net)
    kwargs = {
        "opset_version": 12,
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapper, representative_input(), str(path), **kwargs)
    print("[export]", path)
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print("[onnx] checker passed")
    except ImportError:
        print("[onnx] package unavailable: checker skipped")


def write_yaml(path, model_name, stem, docker_model_root, calibration_dir, calib_count):
    """Write matching eazyai_cvt YAML next to .pth/.onnx.

    Docker is expected to see the project root as /work; model lives under
    /work/ptp/<name>/<name>_pXX_sim.onnx by default.
    """
    yaml = f"""# ACL sparse-pruned {model_name}; CV72 real-data calibration.
# Simplify the ONNX first (python -m onnxsim) before eazyai_cvt.
name: {stem}_c{calib_count}
work_dir: ./out
model_path: {docker_model_root.rstrip('/')}/{stem}_sim.onnx

data_prepare:
  dra_data_1:
    in_path: {calibration_dir.rstrip('/')}/
    in_file_ext: jpg
    out_shape: 1,3,112,1120
    out_data_format: uint8
    transforms:
      - class: GenImageList
        arguments:
          color: RGB

input_nodes:
  input:
    data_prepare: dra_data_1
    mean: 127.5
    std: 127.5

output_nodes:
  output:
    data_format: fp32

cnngen_convert:
  dra_option:
    strategy: auto
"""
    path.write_text(yaml, encoding="utf-8")
    print("[yaml]", path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--name",
        "--model-name",
        dest="name",
        required=True,
        help="Experiment/model directory name, e.g. wacv_ver_mf",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override check_points/<name>/latest_net_G.pth",
    )
    parser.add_argument(
        "--mode", choices=("inspect", "prune", "export"), default="inspect"
    )
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Override ptp/<name>"
    )
    parser.add_argument(
        "--export-from",
        type=Path,
        default=None,
        help="Pruned .pth for export mode; defaults to ptp/<name>/<name>_pXX.pth",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only save .pth during prune; no ONNX or YAML",
    )
    parser.add_argument(
        "--docker-model-root",
        default="/work",
        help="Docker directory containing per-model output folders",
    )
    parser.add_argument(
        "--calibration-dir",
        default="/work/calibration_300",
        help="Docker directory holding calibration JPEGs",
    )
    parser.add_argument(
        "--calib-count",
        type=int,
        default=300,
        help="Suffix for converter job name; does not choose calibration files",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("--name must consist of letters, digits, underscores, or hyphens")
    if not 0 < args.target_sparsity < 1:
        parser.error("--target-sparsity must be in (0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.calib_count < 1:
        parser.error("--calib-count must be positive")

    ckpt = args.checkpoint or Path("check_points") / args.name / "latest_net_G.pth"
    out_dir = args.work_dir or Path("ptp") / args.name
    suffix = f"p{round(args.target_sparsity * 100):02d}"
    stem = f"{args.name}_{suffix}"
    pruned_pth = out_dir / f"{stem}.pth"
    exported_onnx = out_dir / f"{stem}.onnx"
    yaml_path = out_dir / f"{stem}.yaml"
    print(f"[paths] source={ckpt}; output={out_dir}; prefix={stem}")

    net, architecture = make_network(ckpt)
    print("[model] parameters:", sum(p.numel() for p in net.parameters()))
    baseline = report_sparsity(net)
    if args.mode == "inspect":
        print(
            "[inspect] checkpoint matches inferred architecture; ready for L1 pruning"
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        from_pth = args.export_from or pruned_pth
        load_weights(net, read_checkpoint(from_pth), from_pth)
    else:
        fixed = frozen_snapshot(net)
        wrapper = CV72StripWrapper(net).to(args.device)
        acl_l1_prune(wrapper, args.target_sparsity, out_dir, args.device)
        net = wrapper.network.cpu()
        assert_frozen(net, fixed)
        after = report_sparsity(net)
        if after["learnable_sparsity"] <= baseline["learnable_sparsity"] + 0.01:
            raise RuntimeError("ACL did not increase sparsity meaningfully; abort")
        save_checkpoint(
            pruned_pth,
            net,
            args.target_sparsity,
            args.name,
            architecture,
            {"before": baseline, "after": after},
        )
        (out_dir / "sparsity.json").write_text(
            json.dumps(
                {
                    "model_name": args.name,
                    "before": baseline,
                    "after": after,
                    "architecture": architecture,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.no_export:
            return

    export_onnx(net, exported_onnx)
    write_yaml(
        yaml_path,
        args.name,
        stem,
        args.docker_model_root,
        args.calibration_dir,
        args.calib_count,
    )
    print("[next] python -m onnxsim", exported_onnx, out_dir / f"{stem}_sim.onnx")
    print(
        "[next] eazyai_cvt -cy",
        yaml_path,
        "-ac (after copying ptp/ into Docker /work/ptp/)",
    )


if __name__ == "__main__":
    main()

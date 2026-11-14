#!/usr/bin/env python3
"""WACV-6M CV72 ACL PTP pilot: strict checkpoint -> sparse L1 -> ONNX.

Run at the existing WaveBFR project root with models/arch/ available.
Does NOT fine-tune: this is a sparse-acceleration feasibility test, not a quality model.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from models.arch.multi_frame_wavebfr import MultiFrameWaveBFR

EXPECTED_PARAMS = 5_942_338
SHAPE = (1, 3, 112, 1120)
PAD = 128.0 / 127.5 - 1.0


class CV72StripWrapper(nn.Module):
    """Pre-normalized RGB strip [-1,1] -> SR [-1,1]."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network.forward_strip(x)


def load_weights(network, checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full WACV-6M checkpoint not found: {checkpoint}")
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # load only checkpoints you trust
    content = torch.load(str(checkpoint), **kwargs)
    if isinstance(content, nn.Module):
        content = content.state_dict()
    if isinstance(content, dict):
        for k in (
            "state_dict",
            "params_ema",
            "params",
            "model",
            "generator",
            "netG",
            "net_g",
        ):
            if isinstance(content.get(k), dict):
                content = content[k]
                break
    if not isinstance(content, dict):
        raise TypeError(
            "Expected a state_dict or a dict containing state_dict/params/... tensors"
        )
    state = {}
    for name, val in content.items():
        if not torch.is_tensor(val):
            continue
        while name.startswith(("module.", "model.", "network.", "netG.", "net_g.")):
            name = name.split(".", 1)[1]
        if name in state:
            raise RuntimeError(
                f"Duplicate checkpoint key after prefix normalization: {name}"
            )
        state[name] = val
    missing, unexpected = network.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[ERROR] Incomplete checkpoint. Missing:", list(missing)[:20])
        print("[ERROR] Unexpected keys:", list(unexpected)[:20])
        raise RuntimeError(
            "Refusing to prune partially loaded weights. Use the trained multi-frame checkpoint."
        )
    print(f"[weights] STRICT MATCH: {len(state)} tensors from {checkpoint}")


def make_network(checkpoint):
    net = (
        MultiFrameWaveBFR(
            backbone="wavebfr",
            dim=16,
            num_blocks=(2, 2, 2, 4),
            heads=(1, 2, 4, 8),
            num_refinement_blocks=2,
        )
        .eval()
        .float()
    )
    num_params = sum(p.numel() for p in net.parameters())
    if num_params != EXPECTED_PARAMS:
        raise RuntimeError(f"Architecture mismatch: {num_params} != {EXPECTED_PARAMS}")
    load_weights(net, checkpoint)
    return net


def frozen_snapshot(net):
    return {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if not p.requires_grad
    }


def assert_frozen(net, before):
    current = dict(net.named_parameters())
    for n, old in before.items():
        if current[n].requires_grad:
            raise RuntimeError(f"CRITICAL: frozen parameter became trainable: {n}")
        if not torch.equal(current[n].detach().cpu(), old):
            raise RuntimeError(f"CRITICAL: frozen weight was modified by pruning: {n}")
    print(f"[safety] {len(before)} frozen tensors are bitwise unchanged")


def report_sparsity(net):
    frozen, learnable = 0, 0
    zeros_frozen, zeros_learnable = 0, 0
    for _, p in net.named_parameters():
        if p.ndim < 2:
            continue
        count, zero = p.numel(), int((p.detach() == 0).sum().item())
        if p.requires_grad:
            learnable += count
            zeros_learnable += zero
        else:
            frozen += count
            zeros_frozen += zero
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
    # Fixed shape, valid 6-frame prefix, exact neutral-gray padding in [-1,1].
    x = torch.full(SHAPE, PAD, dtype=torch.float32)
    gen = torch.Generator().manual_seed(42)
    x[:, :, :, : 112 * 6] = torch.rand((1, 3, 112, 112 * 6), generator=gen) * 2 - 1
    return x


def verify_strip(net):
    # Wrapper must reproduce explicit-mask forward, else graph is not reliable.
    x = representative_input()
    frames = torch.stack(x.split(112, dim=-1), dim=1)
    mask = torch.arange(10).unsqueeze(0) < 6
    net.eval()
    with torch.inference_mode():
        a = net(frames, frame_mask=mask)
        b = net.forward_strip(x)
    e = float((a - b).abs().max())
    print(f"[strip parity] max abs diff={e:.7g}")
    if not torch.allclose(a, b, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "Explicit-mask and strip wrapper disagree; abort ONNX export"
        )


def acl_l1_prune(wrapper, sparsity, workdir, device):
    os.environ.setdefault("ACL_LOG_DIR", str(workdir / "acl_logs"))
    Path(os.environ["ACL_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import AmbaCompressionLibrary.torch as acl
    except ImportError as e:
        raise RuntimeError(
            "AmbaCompressionLibrary not installed. Use /opt/spongekit/bin/python in Docker."
        ) from e

    # Keyword exclusion is in the vendor guide. Fixed Haar and pixel-unshuffle weights
    # must never change. Depthwise convs are excluded from this speed-only first trial.
    no_prune = ["wavelet_transform", "body.1.conv", "dwconv"]
    print("[ACL] exclude keywords:", no_prune)
    try:
        info = acl.TorchModelInfo(
            wrapper, input_shape=SHAPE, no_prune_keywords=no_prune
        )
    except TypeError as e:
        raise RuntimeError(
            "This ACL TorchModelInfo does not accept no_prune_keywords; inspect your SDK API first."
        ) from e
    algo = acl.algorithm.Uniform("cv72")
    crit = acl.criterion.L1Unstructured()
    schedule = acl.schedule.OneShot(
        target_sparsity=sparsity, sparsity_metric="coefficient"
    )
    pruner = acl.Pruner(info, algo, crit, schedule)
    t = time.time()
    print("[ACL] analyzing 10-slot multi-frame graph; may take time on CPU", flush=True)
    pruner.analyze(verbose=0, export=False)
    masks, weights = pruner.compute_masks(0)
    pruner.apply_masks(wrapper, masks, weights)
    pruner.update_model(wrapper, update_mask=True)
    print(f"[ACL] completed in {time.time() - t:.1f}s")
    try:
        print(
            "[ACL] coefficient sparsity",
            info.coefficient_sparsity(),
            "MAC sparsity",
            info.mac_sparsity(),
        )
    except Exception as exc:
        print("[ACL] optional sparsity report unavailable:", exc)


def save_checkpoint(path, net, sparsity, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only save the model weights; pruning masks are the exact zero entries for later fine-tuning.
    masks = {
        n: (p.detach().cpu() != 0).to(torch.uint8)
        for n, p in net.named_parameters()
        if p.requires_grad and p.ndim >= 2
    }
    torch.save(
        {
            "state_dict": net.state_dict(),
            "masks": masks,
            "meta": {"target_sparsity": sparsity, "architecture": "WACV-6M", **extra},
        },
        path,
    )
    print("[saved]", path)


def export_onnx(net, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Important: ensure fixed PixelUnshuffle and Haar weights are the correct frozen values.
    wrapper = CV72StripWrapper(net.eval().float().cpu())
    verify_strip(net)
    args = {
        "opset_version": 12,
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        args["dynamo"] = False
    torch.onnx.export(wrapper, representative_input(), str(path), **args)
    print("[export]", path)
    try:
        import onnx

        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        print("[onnx] checker passed")
    except ImportError:
        print("[onnx] package unavailable: checker skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="FULL multi-frame training checkpoint (e.g. check_points/wacv_ver_mf/latest_net_G.pth)",
    )
    ap.add_argument("--mode", choices=("inspect", "prune", "export"), default="inspect")
    ap.add_argument("--target-sparsity", type=float, default=0.5)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--work-dir", type=Path, default=Path("runs_wacv6m_ptp"))
    ap.add_argument(
        "--export-from",
        type=Path,
        help="pruned .pth produced by mode=prune; required for mode=export",
    )
    ap.add_argument("--no-export", action="store_true", help="prune and save .pth only")
    args = ap.parse_args()
    if not 0 < args.target_sparsity < 1:
        ap.error("--target-sparsity must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA unavailable in this Python environment")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    net = make_network(args.checkpoint)
    print("[model] parameters:", sum(p.numel() for p in net.parameters()))
    before_stats = report_sparsity(net)
    if args.mode == "inspect":
        print("[inspect] checkpoint matches architecture; ready for L1 pruning")
        return

    if args.mode == "export":
        if args.export_from is None:
            ap.error("--export-from is required for --mode export")
        # Reuse strict load with nested state_dict. Missing tensors cause an error.
        load_weights(net, args.export_from)
    else:
        frozen = frozen_snapshot(net)
        wrapper = CV72StripWrapper(net).to(args.device)
        acl_l1_prune(wrapper, args.target_sparsity, args.work_dir, args.device)
        net = wrapper.network.cpu()
        assert_frozen(net, frozen)
        after_stats = report_sparsity(net)
        if (
            after_stats["learnable_sparsity"]
            <= before_stats["learnable_sparsity"] + 0.01
        ):
            raise RuntimeError("ACL did not increase zero weights meaningfully; abort")
        ckpt = args.work_dir / f"wacv_6m_pruned{int(args.target_sparsity * 100)}.pth"
        save_checkpoint(
            ckpt,
            net,
            args.target_sparsity,
            {"before": before_stats, "after": after_stats},
        )
        (args.work_dir / "sparsity.json").write_text(
            json.dumps({"before": before_stats, "after": after_stats}, indent=2)
        )
        if args.no_export:
            return

    onnx_path = args.work_dir / f"wacv_6m_pruned{int(args.target_sparsity * 100)}.onnx"
    export_onnx(net, onnx_path)
    print("[next] onnxsim + eazyai_cvt -ac + CV72 EVB latency measurement")


if __name__ == "__main__":
    main()

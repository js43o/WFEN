#!/usr/bin/env python3
"""Infer a prepared multi-frame 1120x112 RGB strip with the project PyTorch model.

Optional diagnostics compare the *same normalized input* and the *same loaded
PyTorch checkpoint* against an existing ONNX export and/or an ONNX graph freshly
exported from the in-memory PyTorch network.

Example (from the project's root directory):
    python infer_multi_frame.py \
      --input-image /path/to/19062421_S001_L10_E01_C10.png \
      --output ./results_pytorch/test.png \
      --export-fresh-onnx ./debug/breeze_fresh.onnx \
      --onnx ./breeze_24-224_mf_sim.onnx \
      --pretrain_model_path /path/to/checkpoint.pth \
      --data_device cuda:0 \
      --print-mask

Without --onnx or --export-fresh-onnx, the script simply runs PyTorch inference.
For the comparison or export options, install onnxruntime and onnx as needed.
"""

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from options.test_options import TestOptions
from models import create_model
from utils import utils

HEIGHT = 112
SLOT_WIDTH = 112
SLOTS = 10
WIDTH = SLOT_WIDTH * SLOTS


class CV72StripWrapper(nn.Module):
    """Exactly the project export wrapper, with float RGB [-1,1] input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, strip):
        return self.model.forward_strip(strip)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-image", "--image-path", dest="input_image", type=Path, required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PyTorch PNG path (default: results_pytorch/<stem>.png)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="FP16 PyTorch inference; avoid for FP32 ONNX comparisons",
    )
    parser.add_argument(
        "--print-mask",
        action="store_true",
        help="Print per-slot mean abs and soft validity mask",
    )
    parser.add_argument(
        "--onnx", type=Path, help="Previously exported ONNX to compare against PyTorch"
    )
    parser.add_argument(
        "--export-fresh-onnx",
        type=Path,
        help="Export ONNX from the SAME loaded model, then compare",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("diagnostic_pt_onnx"),
        help="Where raw tensors, comparison PNGs and ONNX outputs are saved",
    )
    args, project_args = parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *project_args]
        opt = TestOptions().parse()
    finally:
        sys.argv = original_argv
    return args, opt


def load_strip(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(
            f"Expected RGB {WIDTH}x{HEIGHT}, got array shape {rgb.shape}; do not resize the 10-frame strip"
        )
    strip = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    strip = strip.div(127.5).sub(1.0)
    return strip


def print_mask(strip):
    print("[Mask] expected convention: padding RGB=128 -> normalized ~0.003922")
    for i in range(SLOTS):
        s = strip[:, :, :, i * SLOT_WIDTH : (i + 1) * SLOT_WIDTH].abs().mean().item()
        x = (s - 0.02) * 100.0
        mask = max(x, 0.0) - max(x - 1.0, 0.0)
        print(f"  slot={i:2d} abs_mean={s:.7f} relu_mask={mask:.7f}")


def load_network(opt, device):
    model = create_model(opt)
    if getattr(opt, "pretrain_model_path", None):
        model.load_pretrain_model()
    else:
        model.setup(opt)
    net = model.netG
    if hasattr(net, "module"):
        net = net.module
    if not hasattr(net, "forward_strip"):
        raise AttributeError(f"{type(net).__name__} has no forward_strip()")
    return net.eval().float().to(device)


def describe(label, x):
    a = np.asarray(x, dtype=np.float64)
    print(
        f"{label}: shape={a.shape} min={a.min():.7f} max={a.max():.7f} "
        f"mean={a.mean():.7f} std={a.std():.7f}"
    )
    if a.ndim == 4 and a.shape[1] == 3:
        print(
            "  channel means (R,G,B):",
            ", ".join(f"{a[:, c].mean():.7f}" for c in range(3)),
        )


def to_png(t):
    """Use the exact original project's postprocessor for both outputs."""
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    arr = np.asarray(utils.tensor_to_img(t, normal=True), dtype=np.uint8)
    if arr.shape != (HEIGHT, SLOT_WIDTH, 3):
        raise ValueError(f"Unexpected tensor_to_img shape {arr.shape}")
    return arr


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def export_fresh(net, strip, output_path):
    """Export the same in-memory FP32 weights using the project's legacy exporter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = CV72StripWrapper(net).eval()
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": 12,
        "export_params": True,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(wrapper, strip, str(output_path), **kwargs)
    print(f"[Export fresh ONNX] {output_path} sha256={sha256(output_path)}")


def compare_onnx(path, strip_nchw, torch_raw, torch_png, dest, tag):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX comparison requires: pip install onnxruntime") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"\n[Compare {tag}] {path} sha256={sha256(path)}")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if len(sess.get_inputs()) != 1 or len(sess.get_outputs()) != 1:
        raise RuntimeError("Expected ONNX with exactly one input and one output")
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(
        f"  ONNX IO: {inp.name} {inp.shape} {inp.type} -> {out.name} {out.shape} {out.type}"
    )
    if inp.type != "tensor(float)":
        raise TypeError(f"Expected float32 ONNX input, got {inp.type}")
    y = sess.run([out.name], {inp.name: strip_nchw})[0]
    if y.shape != torch_raw.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch {torch_raw.shape}, ONNX {y.shape}"
        )
    if not np.isfinite(y).all():
        raise FloatingPointError("ONNX output contains NaN/Inf")
    describe(f"  {tag} ONNX raw", y)
    diff = y.astype(np.float64) - torch_raw.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    maxerr = float(np.max(np.abs(diff)))
    print(f"  RAW diff: MAE={mae:.8f} RMSE={rmse:.8f} MAX={maxerr:.8f}")
    print(
        "  RAW per-channel MAE (R,G,B):",
        ", ".join(f"{np.abs(diff[:, c]).mean():.8f}" for c in range(3)),
    )
    print(
        "  RAW per-channel bias (R,G,B):",
        ", ".join(f"{diff[:, c].mean():+.8f}" for c in range(3)),
    )
    png = to_png(y)
    pixel_error = np.abs(png.astype(np.float32) - torch_png.astype(np.float32))
    print(
        f"  PNG diff: MAE={pixel_error.mean():.4f}/255 MAX={pixel_error.max():.0f}/255"
    )
    Image.fromarray(png, "RGB").save(dest / f"{tag}_onnx.png")
    Image.fromarray(np.concatenate((torch_png, png), axis=1), "RGB").save(
        dest / f"pytorch_left_{tag}_onnx_right.png"
    )
    np.save(dest / f"{tag}_onnx_raw.npy", y)
    return mae, maxerr


def test_fixed_pixel_unshuffle(net):
    """Optional safety check on the already loaded network's non-trainable convolutions."""
    found = 0
    for name, module in net.named_modules():
        if type(module).__name__ != "FixedPixelUnshuffle" or not hasattr(
            module, "conv"
        ):
            continue
        found += 1
        conv = module.conv
        x = torch.randn(
            1,
            conv.in_channels,
            12,
            12,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            actual = module(x)
            expected = torch.nn.functional.pixel_unshuffle(x, 2)
        d = (actual - expected).abs().max().item()
        print(
            f"[FixedPixelUnshuffle] {name}: max error={d:.8g}, weight requires_grad={conv.weight.requires_grad}"
        )
        if d > 1e-6:
            print(
                "  WARNING: This replacement is NOT equivalent to torch.pixel_unshuffle"
            )
    if not found:
        print("[FixedPixelUnshuffle] no such modules found (this is informational)")


def main():
    args, opt = parse_args()
    device = torch.device(opt.data_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")
    if args.fp16 and (args.onnx or args.export_fresh_onnx):
        raise ValueError("Use default FP32 for PyTorch-vs-ONNX comparison; omit --fp16")
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 currently requires CUDA")

    output_path = args.output or Path("results_pytorch") / (
        args.input_image.stem + ".png"
    )
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must be a .png path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strip = load_strip(args.input_image)
    print(f"Input: {args.input_image} RGB {WIDTH}x{HEIGHT}")
    describe("Normalized input [-1,1]", strip.numpy())
    if args.print_mask:
        print_mask(strip)

    net = load_network(opt, device)
    if args.onnx or args.export_fresh_onnx:
        test_fixed_pixel_unshuffle(net)
    if args.fp16:
        net = net.half()
        strip = strip.half()
    net.eval()
    x = strip.to(device)
    with torch.inference_mode():
        torch_output = net.forward_strip(x)
    if isinstance(torch_output, (list, tuple)):
        raise TypeError("Expected one RGB output tensor from forward_strip()")
    if tuple(torch_output.shape) != (1, 3, HEIGHT, SLOT_WIDTH):
        raise ValueError(f"Expected [1,3,112,112], got {tuple(torch_output.shape)}")
    if not torch.isfinite(torch_output).all():
        raise FloatingPointError("PyTorch output contains NaN/Inf")
    raw = torch_output.detach().float().cpu().numpy()
    describe("PyTorch raw output", raw)
    py_png = to_png(torch_output)
    Image.fromarray(py_png, "RGB").save(output_path)
    print(f"Saved PyTorch: {output_path}")

    if args.onnx or args.export_fresh_onnx:
        dest = args.diagnostic_dir
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "pytorch_raw.npy", raw)
        # Use *exactly the same tensor* as the PyTorch call; no second image read.
        input_np = np.ascontiguousarray(strip.float().cpu().numpy(), dtype=np.float32)
        if args.export_fresh_onnx:
            # Fresh ONNX from SAME network and SAME weights, BEFORE simplify.
            export_fresh(net, x, args.export_fresh_onnx)
            compare_onnx(args.export_fresh_onnx, input_np, raw, py_png, dest, "fresh")
        if args.onnx:
            compare_onnx(args.onnx, input_np, raw, py_png, dest, "existing")
        print(f"Diagnostics saved under: {dest.resolve()}")
        print(
            "If fresh ONNX matches but existing ONNX does not, suspect stale weights, wrong export, or simplification."
        )


if __name__ == "__main__":
    main()

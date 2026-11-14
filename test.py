#!/usr/bin/env python3
"""Single-frame WaveBFR evaluation + CV72 ONNX export from the *same* checkpoint.

Run from the original project root, so options/, models/, data/, utils/ resolve.
The project supplies normalized LR tensors. Its existing postprocessor
utils.tensor_to_img(..., normal=True) implies the expected range [-1, 1];
verify this matches the chosen dataset and checkpoint.

Examples:
  python test_single_frame_cv72.py --pretrain_model_path /path/checkpoint.pth \
    --data_device cuda:0 --export-only

  python test_single_frame_cv72.py --pretrain_model_path /path/checkpoint.pth \
    --data_device cuda:0 --calibration-jpg-dir ./calibration_single_300 \
    --export-only

The -sim.onnx file is created later with: python -m onnxsim in.onnx out_sim.onnx
"""

import argparse
import inspect
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils

H = 112
W = 112


class SingleFrameWrapper(nn.Module):
    """ONNX output is always one tensor, including models returning (sr, loss)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        result = self.net(x)
        if isinstance(result, (tuple, list)):
            result = result[0]
        return result


def arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx-out", type=Path, default=Path("wavebfr_single_112.onnx")
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip prediction PNG saving; with --calibration-jpg-dir still build calibration JPGs",
    )
    parser.add_argument(
        "--calibration-jpg-dir",
        type=Path,
        help="Optionally export representative LR JPGs from the dataset used in this invocation",
    )
    parser.add_argument("--calibration-count", type=int, default=300)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--skip-onnx-check", action="store_true")
    parser.add_argument(
        "--compare-onnx",
        type=Path,
        help="Optionally also compare an existing/simplified ONNX against the same PyTorch tensor",
    )
    args, project_argv = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *project_argv]
        opt = TestOptions().parse()
    finally:
        sys.argv = old_argv
    if args.calibration_count < 1:
        parser.error("--calibration-count must be >=1")
    return args, opt


def to_sr_tensor(net, x):
    result = net(x)
    return result[0] if isinstance(result, (tuple, list)) else result


def assert_shape(x, label):
    if tuple(x.shape) != (1, 3, H, W):
        raise RuntimeError(f"{label}: expected [1,3,{H},{W}], got {tuple(x.shape)}")


def compare_onnx(path, inp, torch_output, tag):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Install onnxruntime for ONNX parity checks or use --skip-onnx-check"
        ) from exc
    session = ort.InferenceSession(
        str(path.resolve()), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"{tag} ONNX must have exactly 1 input / 1 output")
    name = inputs[0].name
    xnp = np.ascontiguousarray(inp.detach().cpu().numpy(), dtype=np.float32)
    ynp = session.run([outputs[0].name], {name: xnp})[0]
    ref = torch_output.detach().float().cpu().numpy()
    if ynp.shape != ref.shape:
        raise RuntimeError(f"{tag} output shape mismatch: {ynp.shape} vs {ref.shape}")
    diff = np.abs(ynp.astype(np.float64) - ref.astype(np.float64))
    mae, maximum = float(diff.mean()), float(diff.max())
    print(f"[{tag}] PyTorch vs ONNX: MAE={mae:.8f}, MAX={maximum:.8f}")
    if mae > 1e-3 or maximum > 1e-2:
        raise RuntimeError(
            f"{tag} ONNX parity FAILED; do not convert this model to CV72"
        )
    return ynp


def export_onnx(net, example, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SingleFrameWrapper(net).eval()
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": 12,
        "export_params": True,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(dest), **kwargs)
    print(f"ONNX saved: {dest} | input/output [1,3,{H},{W}]")


def eval_destination(opt):
    if getattr(opt, "save_as_dir", ""):
        return Path(opt.save_as_dir)
    dest = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"
    if getattr(opt, "load_iter", 0) > 0:
        dest = dest.with_name(f"{dest.name}_iter{opt.load_iter}")
    return dest


def normalize_image_and_save(tensor, filename):
    arr = np.asarray(utils.tensor_to_img(tensor, normal=True))
    if arr.shape != (H, W, 3):
        raise RuntimeError(f"Unexpected output PNG shape: {arr.shape}")
    Image.fromarray(arr.astype(np.uint8), "RGB").save(filename)


def main():
    args, opt = arguments()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    dataset = create_dataset(opt)
    iterator = iter(dataset)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Empty evaluation/calibration dataset") from exc

    model = create_model(opt)
    if getattr(opt, "pretrain_model_path", ""):
        model.load_pretrain_model()
    else:
        model.setup(opt)
    net = model.netG.module if isinstance(model.netG, nn.DataParallel) else model.netG
    net.eval().float()
    device = next(net.parameters()).device
    example = first["LR"].to(device=device, dtype=torch.float32)
    assert_shape(example, "LR")
    print(
        f"LR tensor range: {example.min().item():.6f} .. {example.max().item():.6f}; "
        "expected approximately [-1,1]"
    )
    if example.min() < -1.1 or example.max() > 1.1:
        print(
            "[warning] LR values look unlike [-1,1]; confirm dataset normalization before conversion"
        )

    with torch.no_grad():
        ref = to_sr_tensor(net, example)
    assert_shape(ref, "SR")
    export_onnx(net, example, args.onnx_out)
    if not args.skip_onnx_check:
        compare_onnx(args.onnx_out, example, ref, "fresh")
        if args.compare_onnx:
            compare_onnx(args.compare_onnx, example, ref, "existing/simplified")
    if args.export_only and not args.calibration_jpg_dir:
        print("Export-only complete.")
        return

    dest = eval_destination(opt)
    if not args.export_only:
        dest.mkdir(parents=True, exist_ok=True)
        print("Saving model predictions:", dest)
    cal_samples = []
    rng = random.Random(args.calibration_seed)
    seen = 0
    try:
        count = len(dataset)
    except TypeError:
        count = None

    for data in tqdm(
        itertools.chain([first], iterator), total=count, desc="Single frame"
    ):
        lr = data["LR"]
        if tuple(lr.shape) != (1, 3, H, W):
            raise RuntimeError(f"Expected LR [1,3,112,112], got {tuple(lr.shape)}")

        if args.calibration_jpg_dir is not None:
            img = np.asarray(utils.tensor_to_img(lr[0], normal=True), dtype=np.uint8)
            if img.shape != (H, W, 3):
                raise RuntimeError(f"Invalid LR image for calibration: {img.shape}")
            # Reservoir sampling: a uniform sample, not just the earliest IDs.
            if len(cal_samples) < args.calibration_count:
                cal_samples.append((seen, img.copy()))
            else:
                j = rng.randint(0, seen)
                if j < args.calibration_count:
                    cal_samples[j] = (seen, img.copy())

        if not args.export_only:
            with torch.no_grad():
                output = to_sr_tensor(net, lr.to(device=device, dtype=torch.float32))
            assert_shape(output, "SR")
            source_paths = data.get("HR_paths", [])
            basename = (
                Path(str(source_paths[0])).stem if source_paths else f"{seen:06d}"
            )
            normalize_image_and_save(output, dest / f"{basename}.png")
        seen += 1

    if args.calibration_jpg_dir is not None:
        args.calibration_jpg_dir.mkdir(parents=True, exist_ok=True)
        for number, (source_idx, image) in enumerate(
            sorted(cal_samples, key=lambda a: a[0])
        ):
            Image.fromarray(image, "RGB").save(
                args.calibration_jpg_dir / f"{number:06d}_source{source_idx:06d}.jpg",
                quality=100,
                subsampling=0,
            )
        print(
            f"Calibration: {len(cal_samples)} JPGs saved in {args.calibration_jpg_dir}"
        )
        print(
            "[note] Prefer a calibration split with IDs separate from final evaluation IDs."
        )
    if not args.export_only:
        print(f"Saved {seen} predictions in {dest}")


if __name__ == "__main__":
    main()

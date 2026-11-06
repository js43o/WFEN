import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils

WARMUP_ITERS = 100
MEASURE_ITERS = 1000


class CV72StripWrapper(nn.Module):
    """Normalized RGB strip [1,3,112,1120] -> SR [1,3,112,112]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, strip):
        return self.model.forward_strip(strip)


class FlopWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames, mask):
        return self.model(frames, frame_mask=mask)


def check_wrapper(model, device, image_size):
    wrapper = CV72StripWrapper(model).eval()
    with torch.inference_mode():
        for n in (5, 7, 10):
            frames = torch.full(
                (1, 10, 3, image_size, image_size), 128 / 127.5 - 1, device=device
            )
            frames[:, :n] = torch.rand_like(frames[:, :n]) * 2 - 1
            mask = torch.arange(10, device=device)[None, :] < n
            expected = model(frames, frame_mask=mask)
            strip = torch.cat(frames.unbind(1), dim=-1)
            actual = wrapper(strip)
            error = (expected - actual).abs().max().item()
            print(f"Strip wrapper parity ({n} frames): max error={error:.3e}")
            if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-4):
                raise RuntimeError(f"Strip wrapper mismatch ({n} frames)")
    return wrapper


def export_onnx(network, save_path, device, image_size=112):
    model = (
        (network.module if hasattr(network, "module") else network)
        .eval()
        .float()
        .to(device)
    )
    wrapper = check_wrapper(model, device, image_size)
    dummy = torch.full(
        (1, 3, image_size, image_size * 10), 128 / 127.5 - 1, device=device
    )
    dummy[:, :, :, : image_size * 6] = (
        torch.rand_like(dummy[:, :, :, : image_size * 6]) * 2 - 1
    )

    import inspect

    kwargs = {"input_names": ["input"], "output_names": ["output"], "opset_version": 12}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapper, dummy, save_path, **kwargs)
    print(f"ONNX exported: {save_path} | input=[1,3,{image_size},{image_size * 10}]")


def save_calibration_jpgs(dataset, destination):
    """Export unnormalized RGB JPG strips with neutral-gray padding."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(tqdm(dataset, desc="Calibration JPG")):
        frames, mask = data["LR"][0], data["LR_mask"][0].bool()
        strip = utils.tensor_to_img(
            torch.cat(frames.unbind(0), dim=-1), normal=True
        ).copy()
        width = frames.shape[-1]
        for t, valid in enumerate(mask.tolist()):
            if not valid:
                strip[:, t * width : (t + 1) * width, :] = 128
        Image.fromarray(strip).convert("RGB").save(
            destination / f"{i:06d}_{int(mask.sum())}.jpg",
            quality=100,
            subsampling=0,
        )
    print(f"Calibration images saved: {destination}")


def benchmark(network, dataset, device, fp16=True):
    model = network.eval().to(
        device=device, dtype=torch.float16 if fp16 else torch.float32
    )
    dtype = next(model.parameters()).dtype
    print(f"GPU: {torch.cuda.get_device_name(device)}, dtype={dtype}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f} M")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    samples = iter(dataset)
    first = next(samples)
    frames = first["LR"].to(device=device, dtype=dtype)
    mask = first["LR_mask"].to(device=device).bool()
    total_flops = None
    try:
        # Fixed ten-slot graph: masked slots do not eliminate the scheduled operators.
        fp32 = model.float()
        total_flops = FlopCountAnalysis(
            FlopWrapper(fp32), (frames.float(), mask)
        ).total()
        print(f"FLOPs per ten-slot inference: {total_flops / 1e9:.4f} G")
    except Exception as exc:
        print(f"FLOPs unavailable: {exc}")
    finally:
        model.to(dtype=dtype)

    with torch.inference_mode():
        for i, data in enumerate(dataset):
            if i >= WARMUP_ITERS:
                break
            model(
                data["LR"].to(device=device, dtype=dtype),
                frame_mask=data["LR_mask"].to(device).bool(),
            )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mb = torch.cuda.memory_allocated(device) / 2**20

    starts, ends, counts = [], [], []
    with torch.inference_mode():
        for i, data in enumerate(dataset):
            if i >= MEASURE_ITERS:
                break
            frames = data["LR"].to(device=device, dtype=dtype)
            mask = data["LR_mask"].to(device).bool()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            model(frames, frame_mask=mask)
            end.record()
            starts.append(start)
            ends.append(end)
            counts.append(int(data["LR_mask"][0].sum().item()))
    torch.cuda.synchronize()
    if not starts:
        raise RuntimeError("Empty benchmark dataset")

    times = [a.elapsed_time(b) for a, b in zip(starts, ends)]
    grouped = defaultdict(list)
    for count, elapsed in zip(counts, times):
        grouped[count].append(elapsed)
    mean = statistics.mean(times)
    print(
        f"Latency: mean={mean:.4f} ms, median={statistics.median(times):.4f} ms, "
        f"std={statistics.stdev(times) if len(times) > 1 else 0:.4f} ms, "
        f"min={min(times):.4f}, max={max(times):.4f}"
    )
    print(f"Throughput: {1000 / mean:.2f} samples/s | samples={len(times)}")
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"Peak VRAM: {peak_mb:.2f} MB (additional={peak_mb - baseline_mb:.2f} MB)")
    for count, values in sorted(grouped.items()):
        print(
            f"{count} frames: n={len(values):4d}, "
            f"mean={statistics.mean(values):.4f} ms, "
            f"median={statistics.median(values):.4f} ms"
        )
    return {"mean_latency_ms": mean, "flops": total_flops, "peak_vram_mb": peak_mb}


def save_predictions(network, dataset, opt, device, fp16=True):
    dtype = torch.float16 if fp16 else torch.float32
    dest = (
        Path(opt.save_as_dir)
        if opt.save_as_dir
        else (Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}")
    )
    if not opt.save_as_dir and opt.load_iter > 0:
        dest = dest.with_name(f"{dest.name}_iter{opt.load_iter}")
    (dest / "lr").mkdir(parents=True, exist_ok=True)
    (dest / "hr").mkdir(parents=True, exist_ok=True)

    network.eval().to(device=device, dtype=dtype)
    with torch.inference_mode():
        for data in tqdm(dataset, total=len(dataset)):
            frames = data["LR"].to(device=device, dtype=dtype)
            mask = data["LR_mask"].to(device).bool()
            sr = network(frames, frame_mask=mask)
            merged = torch.cat(frames[0].unbind(0), dim=-1)
            parts = Path(data["HR_paths"][0]).parts
            depth = {"multi_frame_multipie": 3, "multi_frame_kface": 5}.get(
                opt.dataset_name, 1
            )
            name = "_".join(parts[-depth:])
            # Image.fromarray(utils.tensor_to_img(merged, normal=True)).save(
            #     dest / "lr" / name
            # )
            Image.fromarray(utils.tensor_to_img(sr, normal=True)).save(dest / name)
            # Image.fromarray(utils.tensor_to_img(data["HR"], normal=True)).save(
            #     dest / "hr" / name
            # )
    print(f"Saved predictions: {dest}")


def main():
    opt = TestOptions().parse()
    opt.num_threads, opt.batch_size, opt.serial_batches, opt.no_flip = 1, 1, True, True
    device = torch.device(opt.data_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")

    dataset = create_dataset(opt)
    if opt.calibration_jpg_dir:
        save_calibration_jpgs(dataset, opt.calibration_jpg_dir)
    model = create_model(opt)
    if opt.pretrain_model_path:
        model.load_pretrain_model()
    else:
        model.setup(opt)
    network = model.netG
    export_onnx(network, f"{opt.name}.onnx", device, image_size=112)
    fp16 = not opt.fp32_benchmark
    benchmark(network, dataset, device, fp16=fp16)
    if not opt.stop_after_benchmark:
        save_predictions(network, dataset, opt, device, fp16=fp16)


if __name__ == "__main__":
    main()

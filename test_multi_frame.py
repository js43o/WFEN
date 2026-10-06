import os
import copy
import statistics
from collections import defaultdict

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils

from PIL import Image
from tqdm import tqdm

import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table

# ============================================================
# Benchmark configuration
# ============================================================

WARMUP_ITERS = 100
MEASURE_ITERS = 1000


# ============================================================
# FLOPs wrapper
# ============================================================


class FlopWrapper(torch.nn.Module):
    """
    fvcore FlopCountAnalysis에
    (inp, frame_mask)를 positional arguments로 전달하기 위한 wrapper.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inp, frame_mask):
        output = self.network(
            inp,
            frame_mask=frame_mask,
        )

        if isinstance(output, tuple):
            output = output[0]

        return output


# ============================================================
# Utility
# ============================================================


def export_onnx(
    network,
    save_path,
    device,
    image_size=112,
):
    network = network.eval().float().to(device)

    # ONNX export
    dummy_inp = torch.randn(
        1,
        10,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )

    dummy_mask = torch.tensor(
        [[True, True, True, True, True, False, False, False, False, False]],
        device=device,
        dtype=torch.bool,
    )

    torch.onnx.export(
        network.module,
        (dummy_inp, dummy_mask),
        save_path,
        input_names=["input", "frame_mask"],
        output_names=["output"],
        opset_version=12,
    )

    print(f"Saved ONNX model: {save_path}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_valid_frame_count(frame_mask):
    """
    frame_mask가
        [B, N]
    형태라고 가정.

    batch_size=1이므로 첫 번째 sample의
    실제 유효 frame 개수를 반환.
    """

    return int(frame_mask[0].sum().item())


def measure_flops_for_sample(
    network,
    inp,
    frame_mask,
    print_table=False,
):
    """
    실제 dataset sample의 shape / mask를 사용하여 FLOPs 측정.

    latency용 모델의 dtype을 변경하지 않기 위해
    FP32 deepcopy를 별도로 만들어 tracing한다.
    """

    device = inp.device

    flop_model = copy.deepcopy(network)
    flop_model = flop_model.eval().float().to(device)

    flop_wrapper = FlopWrapper(flop_model)
    flop_wrapper.eval()

    flop_inp = inp.detach().float()
    flop_mask = frame_mask.detach()

    with torch.inference_mode():
        flops_analysis = FlopCountAnalysis(
            flop_wrapper,
            (flop_inp, flop_mask),
        )

        total_flops = flops_analysis.total()

    if print_table:
        print(
            flop_count_table(
                flops_analysis,
                max_depth=3,
            )
        )

    del flop_wrapper
    del flop_model
    del flop_inp
    del flop_mask
    del flops_analysis

    torch.cuda.empty_cache()

    return total_flops


# ============================================================
# Benchmark
# ============================================================


def run_benchmark(
    network,
    dataset,
    device,
    use_fp16=True,
):
    """
    실제 dataset의 multi-frame input을 이용해 benchmark.

    1) warm-up 100 samples
    2) 실제 1000 samples 측정
    3) CUDA Event를 sample별로 기록
    4) 마지막에 한 번 synchronize
    5) frame-count별 latency와 평균 latency 출력
    6) Peak VRAM 출력
    """

    precision_name = "FP16" if use_fp16 else "FP32"

    dtype = torch.float16 if use_fp16 else torch.float32

    # --------------------------------------------------------
    # Model precision
    # --------------------------------------------------------

    network.eval()
    network.to(device=device, dtype=dtype)

    # NVIDIA inference benchmark 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # FP32일 경우 Ampere/Ada Tensor Core TF32 허용
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params, trainable_params = count_parameters(network)

    print("\n" + "=" * 80)
    print("Model Benchmark Configuration")
    print("=" * 80)

    print(f"GPU                 : " f"{torch.cuda.get_device_name(device)}")

    print(f"Precision           : " f"{precision_name}")

    print(
        f"Total parameters    : " f"{total_params:,} " f"({total_params / 1e6:.4f} M)"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,} "
        f"({trainable_params / 1e6:.4f} M)"
    )

    print(f"Warm-up samples     : " f"{WARMUP_ITERS}")

    print(f"Measured samples    : " f"{MEASURE_ITERS}")

    print("=" * 80 + "\n")

    # --------------------------------------------------------
    # FLOPs cache
    #
    # frame 수별 첫 sample에서만 fvcore 실행.
    # 예: 5/6/7/8/9/10 frame
    # --------------------------------------------------------

    flops_by_frames = {}

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_count = 0

    print("Running warm-up...")

    with torch.inference_mode():

        for data in dataset:

            inp = data["LR"].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            frame_mask = (
                data["LR_mask"]
                .to(
                    device=device,
                    non_blocking=True,
                )
                .bool()
            )

            num_frames = get_valid_frame_count(frame_mask)

            # 해당 frame count의 FLOPs를 아직 계산하지 않았다면
            # 실제 sample로 계산
            if num_frames not in flops_by_frames:

                torch.cuda.synchronize()

                total_flops = measure_flops_for_sample(
                    network=network,
                    inp=inp,
                    frame_mask=frame_mask,
                    print_table=False,
                )

                flops_by_frames[num_frames] = total_flops

                print(
                    f"[FLOPs] "
                    f"{num_frames} frames: "
                    f"{total_flops / 1e9:.4f} GFLOPs"
                )

                # FLOPs tracing이 warm-up 상태를 방해하지 않도록
                # 다시 동기화
                torch.cuda.synchronize()

            _ = network(
                inp,
                frame_mask=frame_mask,
            )

            warmup_count += 1

            if warmup_count >= WARMUP_ITERS:
                break

    torch.cuda.synchronize()

    print(f"Warm-up finished: " f"{warmup_count} samples\n")

    # --------------------------------------------------------
    # Peak VRAM reset
    # --------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 모델 자체가 이미 차지하는 VRAM
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated(device)

    # --------------------------------------------------------
    # Latency events
    # --------------------------------------------------------

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_ITERS)]

    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_ITERS)]

    frame_counts = []

    measured_count = 0

    print("Running latency benchmark...")

    # --------------------------------------------------------
    # Dedicated benchmark pass
    #
    # 이미지 저장 / tensor_to_img / PIL 작업이 없음.
    # --------------------------------------------------------

    with torch.inference_mode():

        for data in dataset:

            if measured_count >= MEASURE_ITERS:
                break

            inp = data["LR"].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            frame_mask = (
                data["LR_mask"]
                .to(
                    device=device,
                    non_blocking=True,
                )
                .bool()
            )

            num_frames = get_valid_frame_count(frame_mask)

            # benchmark 중 처음 등장한 frame count라면
            # FLOPs도 계산
            if num_frames not in flops_by_frames:

                torch.cuda.synchronize()

                total_flops = measure_flops_for_sample(
                    network=network,
                    inp=inp,
                    frame_mask=frame_mask,
                    print_table=False,
                )

                flops_by_frames[num_frames] = total_flops

                print(
                    f"[FLOPs] "
                    f"{num_frames} frames: "
                    f"{total_flops / 1e9:.4f} GFLOPs"
                )

                torch.cuda.synchronize()

            frame_counts.append(num_frames)

            start_events[measured_count].record()

            output = network(
                inp,
                frame_mask=frame_mask,
            )

            if isinstance(output, tuple):
                output = output[0]

            end_events[measured_count].record()

            measured_count += 1

    # 모든 GPU inference가 끝난 뒤 한 번만 동기화
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Timing statistics
    # --------------------------------------------------------

    times_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(measured_count)
    ]

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)

    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    min_ms = min(times_ms)
    max_ms = max(times_ms)

    throughput = 1000.0 / mean_ms

    # --------------------------------------------------------
    # Frame statistics
    # --------------------------------------------------------

    mean_frames = statistics.mean(frame_counts)

    frame_latency = defaultdict(list)

    for num_frames, latency in zip(
        frame_counts,
        times_ms,
    ):
        frame_latency[num_frames].append(latency)

    # --------------------------------------------------------
    # FLOPs average
    #
    # 실제 measured sample의 frame 분포로 weighted average.
    # --------------------------------------------------------

    measured_flops = [flops_by_frames[n] for n in frame_counts if n in flops_by_frames]

    average_flops = statistics.mean(measured_flops) if measured_flops else None

    # --------------------------------------------------------
    # VRAM
    # --------------------------------------------------------

    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)

    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    baseline_allocated_mb = baseline_allocated / (1024**2)

    additional_peak_mb = peak_allocated - baseline_allocated_mb

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Benchmark Result")
    print("=" * 80)

    print(f"GPU                     : " f"{torch.cuda.get_device_name(device)}")

    print(f"Precision               : " f"{precision_name}")

    print(f"Measured samples        : " f"{measured_count}")

    print(f"Average input frames    : " f"{mean_frames:.3f}")

    print("-" * 80)

    if average_flops is not None:
        print(f"Average FLOPs/sample    : " f"{average_flops / 1e9:.4f} GFLOPs")

    print("-" * 80)

    print(f"Mean latency            : " f"{mean_ms:.4f} ms/sample")

    print(f"Median latency          : " f"{median_ms:.4f} ms/sample")

    print(f"Std. deviation          : " f"{std_ms:.4f} ms")

    print(f"Min latency             : " f"{min_ms:.4f} ms")

    print(f"Max latency             : " f"{max_ms:.4f} ms")

    print(f"Throughput              : " f"{throughput:.2f} samples/sec")

    print("-" * 80)

    print(f"Model/base allocated    : " f"{baseline_allocated_mb:.2f} MB")

    print(f"Peak allocated VRAM     : " f"{peak_allocated:.2f} MB")

    print(f"Additional peak VRAM    : " f"{additional_peak_mb:.2f} MB")

    print(f"Peak reserved VRAM      : " f"{peak_reserved:.2f} MB")

    print("=" * 80)

    # --------------------------------------------------------
    # Per-frame-count statistics
    # --------------------------------------------------------

    print("\nLatency by number of valid input frames")
    print("-" * 80)

    print(
        f"{'Frames':>8} "
        f"{'Samples':>10} "
        f"{'FLOPs(G)':>12} "
        f"{'Mean(ms)':>12} "
        f"{'Median(ms)':>12}"
    )

    print("-" * 80)

    for num_frames in sorted(frame_latency.keys()):

        values = frame_latency[num_frames]

        frame_mean = statistics.mean(values)
        frame_median = statistics.median(values)

        flops_g = (
            flops_by_frames[num_frames] / 1e9
            if num_frames in flops_by_frames
            else float("nan")
        )

        print(
            f"{num_frames:>8d} "
            f"{len(values):>10d} "
            f"{flops_g:>12.4f} "
            f"{frame_mean:>12.4f} "
            f"{frame_median:>12.4f}"
        )

    print("=" * 80 + "\n")

    return {
        "precision": precision_name,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "std_latency_ms": std_ms,
        "throughput": throughput,
        "average_frames": mean_frames,
        "average_flops": average_flops,
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    opt = TestOptions().parse()

    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    device = torch.device(opt.data_device)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_dataset(opt)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(opt)

    if len(opt.pretrain_model_path):
        model.load_pretrain_model()
    else:
        model.setup(opt)

    network = model.netG
    network.eval()

    export_onnx(
        network=network,
        save_path="wavebfr_multiframe_opset12.onnx",
        device=device,
        image_size=opt.load_size,
    )

    # --------------------------------------------------------
    # FP16 = default
    #
    # --fp32_benchmark를 넣었을 때만 FP32
    # --------------------------------------------------------

    use_fp16 = not opt.fp32_benchmark

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark_result = run_benchmark(
        network=network,
        dataset=dataset,
        device=device,
        use_fp16=use_fp16,
    )

    # --------------------------------------------------------
    # benchmark만 하고 종료
    # --------------------------------------------------------

    if opt.stop_after_benchmark:

        print("Benchmark completed. " "Stopping before full inference.")

        raise SystemExit(0)

    # --------------------------------------------------------
    # Full inference
    # --------------------------------------------------------

    print("\nBenchmark completed. " "Starting full inference...\n")

    # benchmark에서 사용한 precision 그대로 추론
    inference_dtype = torch.float16 if use_fp16 else torch.float32

    # Save directory
    if len(opt.save_as_dir):
        save_dir = opt.save_as_dir

    else:
        save_dir = os.path.join(
            opt.results_dir,
            opt.name,
            "{}_{}".format(
                opt.phase,
                opt.epoch,
            ),
        )

        if opt.load_iter > 0:
            save_dir = "{:s}_iter{:d}".format(
                save_dir,
                opt.load_iter,
            )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(
        os.path.join(save_dir, "lr"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(save_dir, "hr"),
        exist_ok=True,
    )

    print(
        "creating result directory",
        save_dir,
    )

    # --------------------------------------------------------
    # Normal inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for i, data in tqdm(
            enumerate(dataset),
            total=len(dataset),
        ):

            inp = data["LR"].to(
                device=device,
                dtype=inference_dtype,
            )

            frame_mask = (
                data["LR_mask"]
                .to(
                    device=device,
                )
                .bool()
            )

            # GT는 GPU에 올릴 필요 없음
            hr = data["HR"]

            output = network(
                inp,
                frame_mask=frame_mask,
            )

            if isinstance(output, tuple):
                output = output[0]

            merged_inp = torch.cat(
                [frame for frame in inp[0]],
                dim=-1,
            )

            lr_img = utils.tensor_to_img(
                merged_inp,
                normal=True,
            )

            sr_img = utils.tensor_to_img(
                output,
                normal=True,
            )

            hr_img = utils.tensor_to_img(
                hr,
                normal=True,
            )

            img_path = data["HR_paths"]

            if opt.dataset_name == "multi_frame_multipie":

                filename = "_".join(img_path[0].split("/")[-3:])

            elif opt.dataset_name == "multi_frame_kface":

                filename = "_".join(img_path[0].split("/")[-5:])

            else:

                filename = img_path[0].split("/")[-1]

            Image.fromarray(lr_img).save(
                os.path.join(
                    save_dir,
                    "lr",
                    filename,
                )
            )

            Image.fromarray(sr_img).save(
                os.path.join(
                    save_dir,
                    filename,
                )
            )

            Image.fromarray(hr_img).save(os.path.join(save_dir, "hr", filename))

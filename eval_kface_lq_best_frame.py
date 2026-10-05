import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SESSION = "S001"
TARGET_CAMERAS = [
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
]

DEFAULT_GALLERY_LIGHT = "L1"
DEFAULT_GALLERY_EXPRESSION = "E01"
DEFAULT_GALLERY_CAMERA = "C7"

FRAME_SIZE = 112
NUM_SLOTS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate merged 10-frame K-FACE LQ inputs with best-of-frame "
            "face-recognition metrics."
        )
    )

    parser.add_argument("--lq_dir", type=str, required=True)
    parser.add_argument("--hq_dir", type=str, required=True)

    parser.add_argument(
        "--gallery_mode",
        type=str,
        choices=("canonical", "all_hq"),
        default="canonical",
        help=(
            "Gallery protocol. 'canonical' uses one fixed HQ image per PID. "
            "'all_hq' uses every valid HQ image as a separate gallery entry."
        ),
    )

    parser.add_argument(
        "--gallery_light",
        type=str,
        default=DEFAULT_GALLERY_LIGHT,
    )
    parser.add_argument(
        "--gallery_expression",
        type=str,
        default=DEFAULT_GALLERY_EXPRESSION,
    )
    parser.add_argument(
        "--gallery_camera",
        type=str,
        default=DEFAULT_GALLERY_CAMERA,
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pad_threshold",
        type=float,
        default=0.02,
        help=(
            "Padding-frame threshold in normalized [-1, 1] space. "
            "Frames with mean absolute value <= threshold are treated as padding."
        ),
    )

    return parser.parse_args()


def list_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def parse_kface_path(path):
    """
    Supported forms:
      .../19071011/S001/L1/E01/C7.png
      .../19071011_S001_L1_E01_C7.png
    """
    path = Path(path)
    parts = path.parts

    for i in range(len(parts) - 4):
        if (
            re.fullmatch(r"\d{8}", parts[i])
            and parts[i + 1] == SESSION
            and re.fullmatch(r"L(?:[1-9]|1\d|20)", parts[i + 2])
            and re.fullmatch(r"E0[1-3]", parts[i + 3])
            and re.fullmatch(r"C(?:[1-9]|1\d|20)", path.stem)
        ):
            return {
                "pid": parts[i],
                "session": parts[i + 1],
                "light": parts[i + 2],
                "expression": parts[i + 3],
                "camera": path.stem,
            }

    match = re.fullmatch(
        r"(?P<pid>\d{8})[_-](?P<session>S001)[_-]"
        r"(?P<light>L(?:1\d|20|[1-9]))[_-]"
        r"(?P<expression>E0[1-3])[_-]"
        r"(?P<camera>C(?:1\d|20|[1-9]))",
        path.stem,
    )

    return match.groupdict() if match else None


def build_canonical_gallery(
    hq_dir,
    gallery_light,
    gallery_expression,
    gallery_camera,
):
    gallery = {}

    for path in list_images(hq_dir):
        info = parse_kface_path(path)
        if info is None:
            continue

        if (
            info["session"] == SESSION
            and info["light"] == gallery_light
            and info["expression"] == gallery_expression
            and info["camera"] == gallery_camera
        ):
            gallery.setdefault(info["pid"], path)

    gallery_paths = [gallery[pid] for pid in sorted(gallery)]

    if not gallery_paths:
        raise RuntimeError(
            "No canonical gallery images found. "
            "Check HQ directory and gallery conditions."
        )

    return gallery_paths


def build_all_hq_gallery(hq_dir):
    gallery_paths = []

    for path in list_images(hq_dir):
        info = parse_kface_path(path)
        if info is None:
            continue
        if info["session"] != SESSION:
            continue
        gallery_paths.append(path)

    if not gallery_paths:
        raise RuntimeError("No valid HQ gallery images found.")

    return gallery_paths


def recognition_transform():
    return T.Compose(
        [
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


@torch.no_grad()
def extract_gallery_features(
    paths,
    model,
    transform,
    device,
    batch_size,
):
    features = []
    pids = []

    for start in tqdm(
        range(0, len(paths), batch_size),
        desc="Gallery features",
    ):
        batch_paths = paths[start : start + batch_size]

        images = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)

        batch_features = F.normalize(model(images), dim=1)
        features.append(batch_features.cpu())

        pids.extend(parse_kface_path(path)["pid"] for path in batch_paths)

    return torch.cat(features), np.asarray(pids)


def is_padding_frame(frame_rgb, threshold):
    """
    Padding slots originate from a zero tensor in [-1, 1] space.
    After tensor_to_img(normal=True), they become approximately gray 127/128.
    """
    frame = frame_rgb.astype(np.float32) / 255.0
    frame = frame * 2.0 - 1.0
    return float(np.mean(np.abs(frame))) <= threshold


def split_valid_frames(image_path, pad_threshold):
    """
    Returns valid RGB PIL frames from a horizontally merged 1120x112 image.

    Valid frames are assumed to form a contiguous prefix and padding slots
    occupy the right side.
    """
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image)

    height, width = array.shape[:2]

    expected_width = FRAME_SIZE * NUM_SLOTS
    expected_height = FRAME_SIZE

    if width != expected_width or height != expected_height:
        raise ValueError(
            f"{image_path}: expected {expected_width}x{expected_height}, "
            f"got {width}x{height}."
        )

    valid_count = NUM_SLOTS

    # Search from right to left until the first non-padding frame.
    for frame_idx in range(NUM_SLOTS - 1, -1, -1):
        x1 = frame_idx * FRAME_SIZE
        x2 = x1 + FRAME_SIZE
        frame = array[:, x1:x2]

        if is_padding_frame(frame, pad_threshold):
            valid_count -= 1
        else:
            break

    if valid_count <= 0:
        raise RuntimeError(f"No valid frame found in {image_path}")

    frames = []

    for frame_idx in range(valid_count):
        x1 = frame_idx * FRAME_SIZE
        x2 = x1 + FRAME_SIZE
        frame = array[:, x1:x2]

        # Sanity check for unexpected internal padding.
        if is_padding_frame(frame, pad_threshold):
            raise RuntimeError(
                f"Unexpected internal padding at slot {frame_idx} " f"in {image_path}"
            )

        frames.append(Image.fromarray(frame))

    return frames


@torch.no_grad()
def extract_frame_features(
    frames,
    model,
    transform,
    device,
    batch_size,
):
    features = []

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]

        images = torch.stack([transform(frame) for frame in batch_frames]).to(device)

        batch_features = F.normalize(model(images), dim=1)
        features.append(batch_features.cpu())

    return torch.cat(features)


def tar_at_far(scores, labels, far):
    positives = scores[labels == 1]
    negatives = scores[labels == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return float("nan"), float("nan")

    threshold = np.percentile(
        negatives,
        100 * (1 - far),
    )

    tar = np.mean(positives >= threshold)

    return float(tar), float(threshold)


def evaluate_best_of_frames(
    lq_paths,
    gallery_features,
    gallery_pids,
    model,
    transform,
    device,
    batch_size,
    pad_threshold,
):
    """
    For each merged LQ sample:

    1. Extract all valid frames.
    2. Compute frame-to-gallery similarities.
    3. Rank-N:
       - Evaluate every frame independently.
       - The sample is Rank-1/Rank-5 correct if ANY valid frame is correct.
         This is equivalent to choosing the best individual input frame for
         that sample.
    4. Verification:
       - For each gallery entry, take the maximum similarity over valid frames.
       - TAR@FAR is then computed from these best-of-frame similarities.

    This represents a "best individual input frame" baseline.
    """
    rank1_correct = 0
    rank5_correct = 0

    verification_scores = []
    verification_labels = []

    valid_frame_counts = []

    gallery_features = gallery_features.float()

    for lq_path in tqdm(lq_paths, desc="Best-of-frame evaluation"):
        info = parse_kface_path(lq_path)

        if info is None:
            continue

        probe_pid = info["pid"]

        frames = split_valid_frames(
            lq_path,
            pad_threshold=pad_threshold,
        )

        valid_frame_counts.append(len(frames))

        frame_features = extract_frame_features(
            frames,
            model,
            transform,
            device,
            batch_size,
        ).float()

        # [T, G]
        similarities = frame_features @ gallery_features.T
        similarities_np = similarities.numpy()

        # ------------------------------------------------------------
        # Rank-N
        # ------------------------------------------------------------
        # Evaluate each valid frame independently and keep the best
        # outcome for the sample.
        k = min(5, len(gallery_pids))
        topk_indices = torch.topk(
            similarities,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices.numpy()

        frame_rank1_correct = []
        frame_rank5_correct = []

        for frame_idx in range(len(frames)):
            topk_pids = gallery_pids[topk_indices[frame_idx]]

            frame_rank1_correct.append(bool(topk_pids[0] == probe_pid))
            frame_rank5_correct.append(bool(probe_pid in topk_pids))

        rank1_correct += int(any(frame_rank1_correct))
        rank5_correct += int(any(frame_rank5_correct))

        # ------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------
        # For every gallery entry, keep the highest similarity obtained
        # by any valid frame.
        best_similarities = similarities_np.max(axis=0)

        labels = (gallery_pids == probe_pid).astype(np.int32)

        verification_scores.append(best_similarities)
        verification_labels.append(labels)

    if not valid_frame_counts:
        raise RuntimeError("No valid LQ samples were evaluated.")

    scores = np.concatenate(verification_scores)
    labels = np.concatenate(verification_labels)

    num_samples = len(valid_frame_counts)

    rank1 = rank1_correct / num_samples
    rank5 = rank5_correct / num_samples

    results = {
        "Samples": num_samples,
        "Average_frames": float(np.mean(valid_frame_counts)),
        "Min_frames": int(np.min(valid_frame_counts)),
        "Max_frames": int(np.max(valid_frame_counts)),
        "Rank-1": float(rank1),
        "Rank-5": float(rank5),
    }

    for name, far in (
        ("1e-2", 1e-2),
        ("1e-3", 1e-3),
        ("1e-4", 1e-4),
    ):
        tar, threshold = tar_at_far(
            scores,
            labels,
            far,
        )
        results[f"TAR@FAR={name}"] = tar
        results[f"Threshold@FAR={name}"] = threshold

    return results


def main():
    args = parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    transform = recognition_transform()

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    if args.gallery_mode == "canonical":
        gallery_paths = build_canonical_gallery(
            args.hq_dir,
            args.gallery_light,
            args.gallery_expression,
            args.gallery_camera,
        )
    else:
        gallery_paths = build_all_hq_gallery(
            args.hq_dir,
        )

    gallery_features, gallery_pids = extract_gallery_features(
        gallery_paths,
        model,
        transform,
        device,
        args.batch_size,
    )

    gallery_pid_set = set(gallery_pids.tolist())

    lq_paths = []

    for path in list_images(args.lq_dir):
        info = parse_kface_path(path)

        if (
            info is not None
            and info["pid"] in gallery_pid_set
            and info["camera"] in TARGET_CAMERAS
        ):
            lq_paths.append(path)

    if not lq_paths:
        raise RuntimeError("No valid merged LQ images found.")

    print("\n" + "=" * 70)
    print("K-FACE Best-of-Frame Recognition Evaluation")
    print("=" * 70)
    print(f"LQ directory          : {args.lq_dir}")
    print(f"HQ gallery directory  : {args.hq_dir}")
    print(f"Gallery mode          : {args.gallery_mode}")

    if args.gallery_mode == "canonical":
        print(
            "Gallery condition     : "
            f"{args.gallery_light} / "
            f"{args.gallery_expression} / "
            f"{args.gallery_camera}"
        )

    print(f"Gallery images        : {len(gallery_paths)}")
    print(f"Gallery identities    : {len(set(gallery_pids.tolist()))}")
    print(f"LQ samples            : {len(lq_paths)}")
    print(f"Padding threshold     : {args.pad_threshold}")
    print("=" * 70)

    results = evaluate_best_of_frames(
        lq_paths=lq_paths,
        gallery_features=gallery_features,
        gallery_pids=gallery_pids,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        pad_threshold=args.pad_threshold,
    )

    print("\n" + "=" * 70)
    print("FACE RECOGNITION — BEST INDIVIDUAL INPUT FRAME")
    print("=" * 70)
    print(f"Samples       : {results['Samples']}")
    print(
        f"Valid frames  : avg={results['Average_frames']:.2f}, "
        f"min={results['Min_frames']}, "
        f"max={results['Max_frames']}"
    )

    for far in ("1e-2", "1e-3", "1e-4"):
        print(
            f"TAR @ FAR={far:<4}: "
            f"{results[f'TAR@FAR={far}'] * 100:.2f}% "
            f"(thr={results[f'Threshold@FAR={far}']:.4f})"
        )

    print(f"Rank-1        : {results['Rank-1'] * 100:.2f}%")
    print(f"Rank-5        : {results['Rank-5'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

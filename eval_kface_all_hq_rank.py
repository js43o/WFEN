import argparse
import re
import shutil
import tempfile
from pathlib import Path

import face_alignment
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from pyiqa import create_metric
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

from helpers.arcface.models import resnet_face18
from helpers.utils import process_arcface_input

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified K-FACE image-quality and face-recognition evaluation."
    )
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--hq_dir", type=str, required=True)
    parser.add_argument(
        "--arcface_weights",
        type=str,
        default="helpers/arcface/weights/resnet18_110_wo_dist.pth",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--gallery_light", type=str, default=DEFAULT_GALLERY_LIGHT)
    parser.add_argument(
        "--gallery_expression",
        type=str,
        default=DEFAULT_GALLERY_EXPRESSION,
    )
    parser.add_argument("--gallery_camera", type=str, default=DEFAULT_GALLERY_CAMERA)
    parser.add_argument(
        "--rank_gallery_mode",
        type=str,
        choices=("all_hq", "canonical"),
        default="all_hq",
        help=(
            "Gallery protocol for Rank-N identification. "
            "'all_hq' uses every valid HQ image as gallery; "
            "'canonical' uses one fixed gallery image per PID."
        ),
    )
    parser.add_argument(
        "--match_only",
        action="store_true",
        help=(
            "For image-quality metrics, evaluate only samples that exist in both "
            "pred_dir and hq_dir. Unmatched files on either side are ignored."
        ),
    )
    return parser.parse_args()


def list_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
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

    match = re.search(
        r"(?P<pid>\d{8})[_-](?P<session>S001)[_-]"
        r"(?P<light>L(?:1\d|20|[1-9]))[_-]"
        r"(?P<expression>E0[1-3])[_-]"
        r"(?P<camera>C(?:1\d|20|[1-9]))$",
        path.stem,
    )
    return match.groupdict() if match else None


def sample_key(path):
    info = parse_kface_path(path)
    if info is None:
        return None
    return (
        info["pid"],
        info["session"],
        info["light"],
        info["expression"],
        info["camera"],
    )


def build_image_map(root):
    image_map = {}
    for path in list_images(root):
        key = sample_key(path)
        if key is None:
            continue
        if key in image_map:
            raise RuntimeError(
                f"Duplicate K-FACE sample key found:\n"
                f"  {image_map[key]}\n"
                f"  {path}"
            )
        image_map[key] = path
    return image_map


def prepare_quality_pairs(pred_dir, hq_dir, match_only=False):
    pred_map = build_image_map(pred_dir)
    gt_map = build_image_map(hq_dir)

    if not pred_map:
        raise RuntimeError(f"No valid prediction images found in {pred_dir}")

    pred_keys = set(pred_map)
    gt_keys = set(gt_map)
    matched_keys = sorted(pred_keys & gt_keys)
    pred_only_keys = sorted(pred_keys - gt_keys)
    gt_only_keys = sorted(gt_keys - pred_keys)

    if not match_only and pred_only_keys:
        raise RuntimeError(
            f"{len(pred_only_keys)} prediction samples have no matching HQ GT. "
            f"First missing key: {pred_only_keys[0]}. "
            "Use --match_only to ignore unmatched samples and evaluate only the intersection."
        )

    if not matched_keys:
        raise RuntimeError("No matched prediction/HQ samples were found.")

    print("\nQuality matching")
    print("-" * 70)
    print(f"Available HQ samples : {len(gt_map)}")
    print(f"Prediction samples   : {len(pred_map)}")
    print(f"Matched samples      : {len(matched_keys)}")
    print(f"Prediction only      : {len(pred_only_keys)}")
    print(f"HQ only              : {len(gt_only_keys)}")
    print(f"Match-only mode      : {match_only}")

    return [(pred_map[key], gt_map[key], key) for key in matched_keys]


def load_quality_tensor(path, device):
    return (
        to_tensor(
            Image.open(path)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        .unsqueeze(0)
        .to(device)
    )


def compute_image_quality_metrics(
    pred_dir,
    hq_dir,
    device,
    arcface_weights,
    match_only=False,
):
    pairs = prepare_quality_pairs(
        pred_dir,
        hq_dir,
        match_only=match_only,
    )

    print("\nLoading image-quality metrics...")

    get_psnr = create_metric("psnr", device=device)
    get_ssim = create_metric("ssim", device=device)
    get_lpips = create_metric("lpips", device=device)
    get_niqe = create_metric("niqe", device=device)
    get_fid = create_metric("fid", device=device)
    get_musiq = create_metric("musiq", device=device)
    get_vif = create_metric("vif", device=device)

    id_model = resnet_face18(use_se=False).to(device)
    id_model.load_state_dict(
        torch.load(
            arcface_weights,
            map_location=device,
            weights_only=False,
        )
    )
    id_model.requires_grad_(False)
    id_model.eval()

    landmarks_detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=str(device),
    )

    totals = {
        "PSNR": 0.0,
        "SSIM": 0.0,
        "LPIPS": 0.0,
        "NIQE": 0.0,
        "MUSIQ": 0.0,
        "IDS": 0.0,
        "LMD": 0.0,
        "VIF": 0.0,
    }
    lmd_count = 0

    with torch.no_grad():
        for pred_path, gt_path, _ in tqdm(pairs, desc="Image quality"):
            pred_image = load_quality_tensor(pred_path, device)
            gt_image = load_quality_tensor(gt_path, device)

            pred_feature = F.normalize(
                id_model(process_arcface_input(pred_image)),
                dim=1,
            )
            gt_feature = F.normalize(
                id_model(process_arcface_input(gt_image)),
                dim=1,
            )
            ids_score = F.cosine_similarity(
                pred_feature,
                gt_feature,
                dim=1,
            ).item()

            pred_lds = landmarks_detector.get_landmarks(str(pred_path))
            gt_lds = landmarks_detector.get_landmarks(str(gt_path))

            if pred_lds is not None and gt_lds is not None:
                pred_landmarks = np.asarray(pred_lds[0])
                gt_landmarks = np.asarray(gt_lds[0])
                lmd_score = np.linalg.norm(
                    pred_landmarks - gt_landmarks,
                    axis=1,
                ).mean()
                totals["LMD"] += float(lmd_score)
                lmd_count += 1

            totals["PSNR"] += get_psnr(gt_image, pred_image).item()
            totals["SSIM"] += get_ssim(gt_image, pred_image).item()
            totals["LPIPS"] += get_lpips(gt_image, pred_image).item()
            totals["NIQE"] += get_niqe(pred_image).item()
            totals["MUSIQ"] += get_musiq(pred_image).item()
            totals["VIF"] += get_vif(gt_image, pred_image).item()
            totals["IDS"] += ids_score

    num_samples = len(pairs)
    results = {
        "PSNR": totals["PSNR"] / num_samples,
        "SSIM": totals["SSIM"] / num_samples,
        "LPIPS": totals["LPIPS"] / num_samples,
        "NIQE": totals["NIQE"] / num_samples,
        "MUSIQ": totals["MUSIQ"] / num_samples,
        "IDS": totals["IDS"] / num_samples,
        "LMD": totals["LMD"] / lmd_count if lmd_count > 0 else float("nan"),
        "VIF": totals["VIF"] / num_samples,
        "LMD_valid": lmd_count,
        "Samples": num_samples,
    }

    # FID should use exactly the same matched subset.
    with tempfile.TemporaryDirectory(prefix="kface_fid_") as temp_root:
        temp_root = Path(temp_root)
        temp_gt = temp_root / "gt"
        temp_pred = temp_root / "pred"
        temp_gt.mkdir()
        temp_pred.mkdir()

        for index, (pred_path, gt_path, _) in enumerate(pairs):
            filename = f"{index:08d}.png"
            shutil.copy2(gt_path, temp_gt / filename)
            shutil.copy2(pred_path, temp_pred / filename)

        fid = get_fid(str(temp_gt), str(temp_pred))
        results["FID"] = float(fid.item())

    return results


def build_gallery(hq_dir, gallery_light, gallery_expression, gallery_camera):
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
            "No gallery images found. Check HQ directory and gallery conditions."
        )

    return gallery_paths


def build_all_hq_gallery(hq_dir):
    """
    Build a multi-sample gallery from every valid HQ image.

    Each HQ image is treated as an individual gallery entry, so one PID may
    appear many times in the gallery. This reproduces the previous all-HQ
    identification protocol.
    """
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
def extract_recognition_features(paths, model, transform, device, batch_size):
    features = []
    pids = []

    for start in tqdm(
        range(0, len(paths), batch_size),
        desc="Recognition features",
        leave=False,
    ):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)

        batch_features = F.normalize(model(images), dim=1)
        features.append(batch_features.cpu())
        pids.extend(parse_kface_path(path)["pid"] for path in batch_paths)

    if not features:
        return None, None

    return torch.cat(features), np.asarray(pids)


def verification_accuracy(scores, labels, num_thresholds=1000):
    thresholds = np.linspace(-1, 1, num_thresholds)
    accuracies = ((scores[None] >= thresholds[:, None]) == labels[None]).mean(axis=1)
    index = accuracies.argmax()
    return float(accuracies[index]), float(thresholds[index])


def tar_at_far(scores, labels, far):
    positives = scores[labels == 1]
    negatives = scores[labels == 0]

    if not len(positives) or not len(negatives):
        return float("nan"), float("nan")

    threshold = np.percentile(negatives, 100 * (1 - far))
    tar = (positives >= threshold).mean()
    return float(tar), float(threshold)


def verification_metrics(similarities, probe_pids, gallery_pids):
    labels = (probe_pids[:, None] == gallery_pids[None]).astype(np.int32).ravel()
    scores = similarities.ravel()

    accuracy, accuracy_threshold = verification_accuracy(scores, labels)
    result = {
        "acc": accuracy,
        "acc_thr": accuracy_threshold,
    }

    for name, far in (("1e2", 1e-2), ("1e3", 1e-3), ("1e4", 1e-4)):
        result[f"tar_{name}"], result[f"thr_{name}"] = tar_at_far(
            scores,
            labels,
            far,
        )

    return result


def identification_metrics_batched(
    probe_features,
    probe_pids,
    gallery_features,
    gallery_pids,
    batch_size,
):
    """
    Compute Rank-1 / Rank-5 without constructing the full
    [num_probe, num_gallery] similarity matrix.

    Gallery entries are ranked image-by-image, matching the previous
    all-HQ gallery protocol. If multiple HQ images belong to the same PID,
    they occupy separate gallery positions.
    """
    rank1 = 0
    rank5 = 0
    num_probes = len(probe_pids)
    k = min(5, len(gallery_pids))

    gallery_features = gallery_features.float()

    for start in tqdm(
        range(0, num_probes, batch_size),
        desc="Identification",
        leave=False,
    ):
        end = min(start + batch_size, num_probes)
        batch_features = probe_features[start:end].float()

        similarities = batch_features @ gallery_features.T
        topk_indices = (
            torch.topk(
                similarities,
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            )
            .indices.cpu()
            .numpy()
        )

        for local_idx, probe_idx in enumerate(range(start, end)):
            pid = probe_pids[probe_idx]
            topk_pids = gallery_pids[topk_indices[local_idx]]

            rank1 += int(topk_pids[0] == pid)
            rank5 += int(pid in topk_pids)

    return rank1 / num_probes, rank5 / num_probes


def compute_recognition_metrics(
    pred_dir,
    hq_dir,
    device,
    batch_size,
    gallery_light,
    gallery_expression,
    gallery_camera,
    rank_gallery_mode,
):
    transform = recognition_transform()
    id_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Verification keeps the existing one-image-per-PID canonical gallery.
    verification_gallery_paths = build_gallery(
        hq_dir,
        gallery_light,
        gallery_expression,
        gallery_camera,
    )

    print("\nFace-recognition setup")
    print("-" * 70)
    print(
        "Verification gallery  : "
        f"{gallery_light} / {gallery_expression} / {gallery_camera}"
    )
    print(f"Gallery identities    : {len(verification_gallery_paths)}")

    verification_gallery_features, verification_gallery_pids = (
        extract_recognition_features(
            verification_gallery_paths,
            id_model,
            transform,
            device,
            batch_size,
        )
    )

    gallery_pid_set = set(verification_gallery_pids.tolist())
    probe_paths = []

    for path in list_images(pred_dir):
        info = parse_kface_path(path)
        if (
            info is not None
            and info["pid"] in gallery_pid_set
            and info["camera"] in TARGET_CAMERAS
        ):
            probe_paths.append(path)

    if not probe_paths:
        raise RuntimeError("No valid probe images found.")

    print(f"Probe images          : {len(probe_paths)}")

    probe_features, probe_pids = extract_recognition_features(
        probe_paths,
        id_model,
        transform,
        device,
        batch_size,
    )

    # Verification: unchanged canonical one-shot gallery.
    verification_similarities = (
        probe_features @ verification_gallery_features.T
    ).numpy()

    verification = verification_metrics(
        verification_similarities,
        probe_pids,
        verification_gallery_pids,
    )

    # Identification: either the previous all-HQ gallery protocol or
    # the newer one-image-per-PID canonical gallery protocol.
    if rank_gallery_mode == "all_hq":
        rank_gallery_paths = build_all_hq_gallery(hq_dir)
        print(f"Rank-N gallery mode   : all HQ images")
        print(f"Rank-N gallery images : {len(rank_gallery_paths)}")

        rank_gallery_features, rank_gallery_pids = extract_recognition_features(
            rank_gallery_paths,
            id_model,
            transform,
            device,
            batch_size,
        )
    else:
        rank_gallery_paths = verification_gallery_paths
        rank_gallery_features = verification_gallery_features
        rank_gallery_pids = verification_gallery_pids
        print(f"Rank-N gallery mode   : canonical one-per-PID")
        print(f"Rank-N gallery images : {len(rank_gallery_paths)}")

    rank1, rank5 = identification_metrics_batched(
        probe_features=probe_features,
        probe_pids=probe_pids,
        gallery_features=rank_gallery_features,
        gallery_pids=rank_gallery_pids,
        batch_size=batch_size,
    )

    return {
        "Accuracy": verification["acc"],
        "Accuracy_threshold": verification["acc_thr"],
        "TAR@FAR=1e-2": verification["tar_1e2"],
        "Threshold@FAR=1e-2": verification["thr_1e2"],
        "TAR@FAR=1e-3": verification["tar_1e3"],
        "Threshold@FAR=1e-3": verification["thr_1e3"],
        "TAR@FAR=1e-4": verification["tar_1e4"],
        "Threshold@FAR=1e-4": verification["thr_1e4"],
        "Rank-1": float(rank1),
        "Rank-5": float(rank5),
        "Gallery_size": len(verification_gallery_pids),
        "Probe_size": len(probe_pids),
        "Rank_gallery_size": len(rank_gallery_pids),
        "Rank_gallery_mode": rank_gallery_mode,
    }


def print_results(quality, recognition):
    print("\n" + "=" * 70)
    print("IMAGE QUALITY")
    print("=" * 70)
    print(f"Samples       : {quality['Samples']}")
    print(f"PSNR          : {quality['PSNR']:.6f}")
    print(f"SSIM          : {quality['SSIM']:.6f}")
    print(f"LPIPS         : {quality['LPIPS']:.6f}")
    print(f"NIQE          : {quality['NIQE']:.6f}")
    print(f"MUSIQ         : {quality['MUSIQ']:.6f}")
    print(f"FID           : {quality['FID']:.6f}")
    print(f"IDS           : {quality['IDS']:.6f}")
    print(
        f"LMD           : {quality['LMD']:.6f} "
        f"({quality['LMD_valid']}/{quality['Samples']} valid)"
    )
    print(f"VIF           : {quality['VIF']:.6f}")

    print("\n" + "=" * 70)
    print("FACE RECOGNITION")
    print("=" * 70)
    print(f"Gallery size  : {recognition['Gallery_size']}")
    print(f"Probe size    : {recognition['Probe_size']}")
    print(
        f"Rank gallery  : {recognition['Rank_gallery_mode']} "
        f"({recognition['Rank_gallery_size']} images)"
    )
    print(
        f"Accuracy      : {recognition['Accuracy'] * 100:.2f}% "
        f"(thr={recognition['Accuracy_threshold']:.4f})"
    )

    for far in ("1e-2", "1e-3", "1e-4"):
        print(
            f"TAR @ FAR={far:<4}: "
            f"{recognition[f'TAR@FAR={far}'] * 100:.2f}% "
            f"(thr={recognition[f'Threshold@FAR={far}']:.4f})"
        )

    print(f"Rank-1        : {recognition['Rank-1'] * 100:.2f}%")
    print(f"Rank-5        : {recognition['Rank-5'] * 100:.2f}%")
    print("=" * 70)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print("=" * 70)
    print("Unified K-FACE Evaluation")
    print("=" * 70)
    print(f"Prediction dir : {args.pred_dir}")
    print(f"HQ dir         : {args.hq_dir}")
    print(f"Device         : {device}")
    print(f"Match only     : {args.match_only}")
    print("=" * 70)

    quality = compute_image_quality_metrics(
        pred_dir=args.pred_dir,
        hq_dir=args.hq_dir,
        device=device,
        arcface_weights=args.arcface_weights,
        match_only=args.match_only,
    )

    recognition = compute_recognition_metrics(
        pred_dir=args.pred_dir,
        hq_dir=args.hq_dir,
        device=device,
        batch_size=args.batch_size,
        gallery_light=args.gallery_light,
        gallery_expression=args.gallery_expression,
        gallery_camera=args.gallery_camera,
        rank_gallery_mode=args.rank_gallery_mode,
    )

    print_results(quality, recognition)


if __name__ == "__main__":
    main()

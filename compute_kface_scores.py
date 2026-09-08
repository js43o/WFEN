import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm


GT_DIR = "../../datasets/kface_crop_patch_v2/test"
MODEL_ROOT = "results"
MODEL_DIRS = ["breeze_mf_v3/kface_crop_patch_v2"]

SESSION = "S001"
TARGET_CAMERAS = [
    "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C14", "C15", "C16", "C17", "C18",
]

# PID당 gallery 한 장. 필요하면 조건만 바꾸면 됨.
GALLERY_LIGHT = "L1"
GALLERY_EXPRESSION = "E01"
GALLERY_CAMERA = "C7"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


id_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def parse_kface_path(path):
    """
    다음 두 형태를 지원:
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
        r"(?P<light>L(?:[1-9]|1\d|20))[_-]"
        r"(?P<expression>E0[1-3])[_-]"
        r"(?P<camera>C(?:[1-9]|1\d|20))",
        path.stem,
    )
    return match.groupdict() if match else None


def list_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_image(path):
    return transform(Image.open(path).convert("RGB"))


@torch.no_grad()
def extract_features(paths):
    features, pids = [], []

    for start in tqdm(range(0, len(paths), BATCH_SIZE), leave=False):
        batch_paths = paths[start:start + BATCH_SIZE]
        images = torch.stack([load_image(p) for p in batch_paths]).to(DEVICE)
        features.append(F.normalize(id_model(images), dim=1).cpu())
        pids.extend(parse_kface_path(p)["pid"] for p in batch_paths)

    if not features:
        return None, None
    return torch.cat(features), np.asarray(pids)


def verification_accuracy(scores, labels, num_thresholds=1000):
    thresholds = np.linspace(-1, 1, num_thresholds)
    accuracies = ((scores[None] >= thresholds[:, None]) == labels[None]).mean(1)
    index = accuracies.argmax()
    return float(accuracies[index]), float(thresholds[index])


def tar_at_far(scores, labels, far):
    positives = scores[labels == 1]
    negatives = scores[labels == 0]

    if not len(positives) or not len(negatives):
        return float("nan"), float("nan")

    threshold = np.percentile(negatives, 100 * (1 - far))
    return float((positives >= threshold).mean()), float(threshold)


def verification_metrics(similarities, probe_pids, gallery_pids):
    labels = (probe_pids[:, None] == gallery_pids[None]).astype(np.int32).ravel()
    scores = similarities.ravel()
    accuracy, accuracy_threshold = verification_accuracy(scores, labels)

    result = {"acc": accuracy, "acc_thr": accuracy_threshold}
    for name, far in (("1e2", 1e-2), ("1e3", 1e-3), ("1e4", 1e-4)):
        result[f"tar_{name}"], result[f"thr_{name}"] = tar_at_far(scores, labels, far)
    return result


def identification_metrics(similarities, probe_pids, gallery_pids):
    order = np.argsort(-similarities, axis=1)
    top1 = gallery_pids[order[:, :1]]
    top5 = gallery_pids[order[:, :min(5, len(gallery_pids))]]

    rank1 = np.mean([pid in row for pid, row in zip(probe_pids, top1)])
    rank5 = np.mean([pid in row for pid, row in zip(probe_pids, top5)])
    return float(rank1), float(rank5)


def build_gallery():
    gallery = {}

    for path in list_images(GT_DIR):
        info = parse_kface_path(path)
        if not info:
            continue
        if (
            info["session"] == SESSION
            and info["light"] == GALLERY_LIGHT
            and info["expression"] == GALLERY_EXPRESSION
            and info["camera"] == GALLERY_CAMERA
        ):
            gallery.setdefault(info["pid"], path)

    return [gallery[pid] for pid in sorted(gallery)]


gallery_paths = build_gallery()
print(f"Building gallery: {len(gallery_paths)} images")

gallery_features, gallery_pids = extract_features(gallery_paths)
if gallery_features is None:
    raise RuntimeError("No gallery images found. Check GT_DIR and gallery conditions.")

print("Gallery size:", len(gallery_pids))


for model_name in MODEL_DIRS:
    print("\n==============================")
    print("Model:", model_name)

    probe_paths = []
    gallery_pid_set = set(gallery_pids)

    for path in list_images(os.path.join(MODEL_ROOT, model_name)):
        info = parse_kface_path(path)
        if info and info["pid"] in gallery_pid_set and info["camera"] in TARGET_CAMERAS:
            probe_paths.append(path)

    print("Probe size:", len(probe_paths))
    if not probe_paths:
        print("No valid probe images. Skipped.")
        continue

    probe_features, probe_pids = extract_features(probe_paths)
    similarities = (probe_features @ gallery_features.T).numpy()

    verification = verification_metrics(similarities, probe_pids, gallery_pids)
    rank1, rank5 = identification_metrics(similarities, probe_pids, gallery_pids)

    print("Verification:")
    print(f"Accuracy          : {verification['acc'] * 100:.2f}  "
          f"(thr={verification['acc_thr']:.4f})")
    for name, label in (("1e2", "1e-2"), ("1e3", "1e-3"), ("1e4", "1e-4")):
        print(f"TAR @ FAR={label:<5} : {verification[f'tar_{name}'] * 100:.2f}  "
              f"(thr={verification[f'thr_{name}']:.4f})")

    print("Identification:")
    print(f"Rank-1            : {rank1 * 100:.2f}")
    print(f"Rank-5            : {rank5 * 100:.2f}")

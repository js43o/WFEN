import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1


# ──────────────────────────────────────────
# Settings
# ──────────────────────────────────────────
GT_DIR = "../../datasets/multipie_validation_v2/gt"
MODEL_ROOT = "results"
MODEL_DIRS = ["breeze_122/multipie_crop_patch_v2"]

ANGLES_MODERATE = [
    "19_1",
    "19_0",
    "04_1",
    "13_0",
    "08_0",
    "08_1",
]

ANGLES_NEAR_FRONTAL = [
    "05_0",
    "05_1",
    "14_0",
]

TARGET_POSES = ANGLES_MODERATE + ANGLES_NEAR_FRONTAL

PIDS = list(range(201, 251))
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────
# Model & preprocessing
# ──────────────────────────────────────────
id_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

transform = T.Compose(
    [
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


# ──────────────────────────────────────────
# Utils
# ──────────────────────────────────────────
def get_pid(filename: str) -> int:
    return int(filename.split("_")[0])


def get_pose(filename: str) -> str:
    parts = filename.split("_")
    return f"{parts[1]}_{parts[2]}"


def load_image(path: str) -> torch.Tensor:
    return transform(Image.open(path).convert("RGB"))


@torch.no_grad()
def extract_features(paths, batch_size=BATCH_SIZE):
    feats = []
    pids = []

    for i in tqdm(range(0, len(paths), batch_size), leave=False):
        batch_paths = paths[i : i + batch_size]
        imgs = torch.stack([load_image(p) for p in batch_paths]).to(DEVICE)

        feat = id_model(imgs)
        feat = torch.nn.functional.normalize(feat, dim=1)

        feats.append(feat.cpu())
        pids.extend([get_pid(os.path.basename(p)) for p in batch_paths])

    if len(feats) == 0:
        return None, None

    return torch.cat(feats, dim=0), np.array(pids)


def compute_verification_accuracy(scores, labels, num_thresholds=1000):
    thresholds = np.linspace(-1.0, 1.0, num_thresholds)
    preds = scores[None, :] >= thresholds[:, None]
    accs = (preds == labels[None, :]).mean(axis=1)

    best_idx = np.argmax(accs)
    return float(accs[best_idx]), float(thresholds[best_idx])


def compute_tar_at_far(scores, labels, target_far):
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    threshold = np.percentile(neg_scores, 100.0 * (1.0 - target_far))
    tar = np.mean(pos_scores >= threshold)

    return float(tar), float(threshold)


def compute_verification_metrics(sims, probe_pids, gallery_pids):
    pair_labels = (probe_pids[:, None] == gallery_pids[None, :]).astype(np.int32)

    scores = sims.reshape(-1)
    labels = pair_labels.reshape(-1)

    acc, acc_thr = compute_verification_accuracy(scores, labels)
    tar_1e2, thr_1e2 = compute_tar_at_far(scores, labels, target_far=1e-2)
    tar_1e3, thr_1e3 = compute_tar_at_far(scores, labels, target_far=1e-3)
    tar_1e4, thr_1e4 = compute_tar_at_far(scores, labels, target_far=1e-4)

    return {
        "acc": acc,
        "acc_thr": acc_thr,
        "tar_1e2": tar_1e2,
        "thr_1e2": thr_1e2,
        "tar_1e3": tar_1e3,
        "thr_1e3": thr_1e3,
        "tar_1e4": tar_1e4,
        "thr_1e4": thr_1e4,
    }


def compute_identification_metrics(sims, probe_pids, gallery_pids):
    order = np.argsort(-sims, axis=1)

    rank1 = 0
    rank5 = 0

    k = min(5, len(gallery_pids))

    for i, pid in enumerate(probe_pids):
        top1 = gallery_pids[order[i, :1]]
        top5 = gallery_pids[order[i, :k]]

        rank1 += int(pid in top1)
        rank5 += int(pid in top5)

    rank1 /= len(probe_pids)
    rank5 /= len(probe_pids)

    return rank1, rank5


# ──────────────────────────────────────────
# Build gallery
# Gallery: frontal GT image, PID_01_0_16.png
# ──────────────────────────────────────────
gallery_paths = []

for pid in PIDS:
    path = os.path.join(GT_DIR, f"{pid:03d}_01_0_16.png")

    if os.path.exists(path):
        gallery_paths.append(path)
    else:
        print("Missing gallery image:", os.path.basename(path))

print(f"Building gallery: {len(gallery_paths)} images")

gallery_feats, gallery_pids = extract_features(gallery_paths)

if gallery_feats is None:
    raise RuntimeError("No gallery images found.")

print("Gallery size:", len(gallery_pids))


# ──────────────────────────────────────────
# Evaluate models
# ──────────────────────────────────────────
for model_name in MODEL_DIRS:
    print("\n==============================")
    print("Model:", model_name)

    model_dir = os.path.join(MODEL_ROOT, model_name)
    all_files = sorted(glob.glob(os.path.join(model_dir, "*.png")))

    probe_paths = []

    for path in all_files:
        filename = os.path.basename(path)
        pid = get_pid(filename)
        pose = get_pose(filename)

        if pid in PIDS and pose in TARGET_POSES:
            probe_paths.append(path)

    print("Probe size:", len(probe_paths))

    if len(probe_paths) == 0:
        print("No valid probe images. Skipped.")
        continue

    probe_feats, probe_pids = extract_features(probe_paths)

    # cosine similarity because features are L2-normalized
    sims = torch.matmul(probe_feats, gallery_feats.T).numpy()

    # Verification
    verif = compute_verification_metrics(sims, probe_pids, gallery_pids)

    # Identification
    rank1, rank5 = compute_identification_metrics(sims, probe_pids, gallery_pids)

    print("Verification:")
    print(f"Accuracy          : {verif['acc'] * 100:.2f}  (thr={verif['acc_thr']:.4f})")
    print(f"TAR @ FAR=1e-2    : {verif['tar_1e2'] * 100:.2f}  (thr={verif['thr_1e2']:.4f})")
    print(f"TAR @ FAR=1e-3    : {verif['tar_1e3'] * 100:.2f}  (thr={verif['thr_1e3']:.4f})")
    print(f"TAR @ FAR=1e-4    : {verif['tar_1e4'] * 100:.2f}  (thr={verif['thr_1e4']:.4f})")

    print("Identification:")
    print(f"Rank-1            : {rank1 * 100:.2f}")
    print(f"Rank-5            : {rank5 * 100:.2f}")

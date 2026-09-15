import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from basicsr.data import degradations
from basicsr.utils import img2tensor
from data.base_dataset import BaseDataset

SESSION = "S001"
LIGHT_CLASSES = ["L1", "L2", "L3", "L4", "L8", "L9", "L10", "L12", "L13"]
EXPRESSION_CLASSES = ["E01", "E02", "E03"]

PITCH_ZERO_CAMERAS = ["C4", "C5", "C6", "C7", "C8", "C9", "C10"]
PITCH_POSITIVE_CAMERAS = ["C14", "C15", "C16", "C17", "C18"]
CAMERA_CLASSES = PITCH_ZERO_CAMERAS + PITCH_POSITIVE_CAMERAS

# 기본적으로 지정한 모든 카메라를 reference/GT 후보로 허용.
GT_CAMERAS = CAMERA_CLASSES


class MultiFrameKFaceDataset(BaseDataset):
    """K-FACE 기반 가변 길이 다중 프레임 얼굴 SR 데이터셋."""

    def __init__(self, opt):
        super().__init__(opt)

        self.img_dir = opt.dataroot
        self.img_size = opt.load_size
        self.min_frames = getattr(opt, "min_frames", 5)
        self.max_frames = getattr(opt, "max_frames", 10)

        self.first_size_range = getattr(opt, "first_lr_size_range", (16, 32))
        self.last_size_range = getattr(opt, "last_lr_size_range", (32, 64))
        self.size_gamma = getattr(opt, "lr_size_gamma", 1.5)
        self.size_jitter = getattr(opt, "lr_size_jitter_ratio", 0.25)

        self.allow_duplicate_frames = getattr(opt, "allow_duplicate_frames", False)

        self.blur_kernel_size = opt.blur_kernel_size
        self.kernel_list = opt.kernel_list
        self.kernel_prob = opt.kernel_prob
        self.blur_sigma = opt.blur_sigma
        self.noise_range = opt.noise_range
        self.jpeg_range = opt.jpeg_range

        self.samples = []
        self.reference_samples = []
        self.samples_by_pid = defaultdict(list)

        self._validate_options()
        self._build_index()

        if not self.reference_samples:
            raise RuntimeError(f"No valid K-FACE samples found in {self.img_dir}")

        print(
            f"[MultiFrameKFaceDataset] {len(self.samples)} images, "
            f"{len(self.reference_samples)} references, "
            f"{len(self.samples_by_pid)} identities"
        )

    def _validate_options(self):
        if not 2 <= self.min_frames <= self.max_frames:
            raise ValueError("Require 2 <= min_frames <= max_frames.")

        for name, value in (
            ("first_lr_size_range", self.first_size_range),
            ("last_lr_size_range", self.last_size_range),
        ):
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError(f"{name} must be (min, max).")

        if self.first_size_range[1] > self.last_size_range[0]:
            raise ValueError(
                "The first-frame range must end below the last-frame range."
            )
        if self.size_gamma <= 0 or self.size_jitter < 0:
            raise ValueError("Invalid size_gamma or size_jitter.")

    def _build_index(self):
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)

        for pid in sorted(os.listdir(self.img_dir)):
            if len(pid) != 8 or not pid.isdigit():
                continue

            session_dir = os.path.join(self.img_dir, pid, SESSION)
            if not os.path.isdir(session_dir):
                continue

            for light in LIGHT_CLASSES:
                for expression in EXPRESSION_CLASSES:
                    for camera in CAMERA_CLASSES:
                        path = os.path.join(
                            session_dir,
                            light,
                            expression,
                            f"{camera}.png",
                        )
                        if not os.path.isfile(path):
                            continue

                        sample = {
                            "pid": pid,
                            "session": SESSION,
                            "light": light,
                            "expression": expression,
                            "camera": camera,
                            "path": path,
                        }
                        self.samples.append(sample)
                        self.samples_by_pid[pid].append(sample)

                        if camera in GT_CAMERAS:
                            self.reference_samples.append(sample)

    @staticmethod
    def _random_interpolation():
        return random.choice((cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC))

    @staticmethod
    def _to_tensor(image, round_8bit=False):
        tensor = img2tensor(image, bgr2rgb=True, float32=True)
        if round_8bit:
            tensor = tensor.mul(255).round().clamp_(0, 255).div_(255)
        normalize(tensor, [0.5] * 3, [0.5] * 3, inplace=True)
        return tensor

    def _read_image(self, path, interpolation):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)

        image = cv2.resize(image, (512, 512), interpolation=interpolation)
        return image.astype(np.float32) / 255.0

    def _sample_auxiliary(self, reference, count):
        candidates = self.samples_by_pid[reference["pid"]]

        if not self.allow_duplicate_frames:
            candidates = [x for x in candidates if x["path"] != reference["path"]]

        candidates = candidates or [reference]

        if len(candidates) >= count:
            return random.sample(candidates, count)
        return random.choices(candidates, k=count)

    def _sample_sizes(self, count):
        first = random.randint(*self.first_size_range)
        last = random.randint(*self.last_size_range)

        t = np.linspace(0, 1, count, dtype=np.float32)
        sizes = first + (last - first) * t**self.size_gamma
        step = (last - first) / max(count - 1, 1)
        sizes += np.random.normal(0, step * self.size_jitter, count)
        sizes[[0, -1]] = first, last

        return np.clip(np.rint(sizes), first, last).astype(np.int64).tolist()

    def _degrade(self, image, size, interpolation):
        image = image.copy()

        if self.blur_kernel_size is not None:
            low, high = self.blur_kernel_size
            if low >= high:
                raise ValueError("blur_kernel_size must satisfy min < max.")

            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                random.randint(low, high) * 2 + 1,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None,
            )
            image = cv2.filter2D(image, -1, kernel)

        image = cv2.resize(image, (size, size), interpolation=interpolation)

        if self.noise_range is not None:
            image = degradations.random_add_gaussian_noise(image, self.noise_range)
        if self.jpeg_range is not None:
            image = degradations.random_add_jpg_compression(image, self.jpeg_range)

        image = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=interpolation,
        )
        return np.clip(image, 0, 1)

    def __getitem__(self, index):
        reference = self.reference_samples[index]
        interpolation = self._random_interpolation()
        num_frames = random.randint(self.min_frames, self.max_frames)

        samples = self._sample_auxiliary(reference, num_frames - 1) + [reference]
        sizes = self._sample_sizes(num_frames)

        reference_image = self._read_image(reference["path"], interpolation)
        reference_image = cv2.resize(
            reference_image,
            (self.img_size, self.img_size),
            interpolation=interpolation,
        )

        lr = torch.zeros(self.max_frames, 3, self.img_size, self.img_size)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_sizes = torch.zeros(self.max_frames, dtype=torch.long)

        paths = [""] * self.max_frames
        lights = [""] * self.max_frames
        expressions = [""] * self.max_frames
        cameras = [""] * self.max_frames

        for i, (sample, size) in enumerate(zip(samples, sizes)):
            image = self._read_image(sample["path"], interpolation)
            lr[i] = self._to_tensor(
                self._degrade(image, size, interpolation),
                round_8bit=True,
            )
            mask[i] = True
            frame_sizes[i] = size
            paths[i] = sample["path"]
            lights[i] = sample["light"]
            expressions[i] = sample["expression"]
            cameras[i] = sample["camera"]

        return {
            "HR": self._to_tensor(reference_image),
            "LR": lr,
            "LR_mask": mask,
            "num_frames": torch.tensor(num_frames),
            "frame_sizes": frame_sizes,
            "HR_paths": reference["path"],
            "LR_paths": paths,
            "lights": lights,
            "expressions": expressions,
            "cameras": cameras,
            "pid": reference["pid"],
        }

    def __len__(self):
        return len(self.reference_samples)

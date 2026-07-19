import os
import random
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from basicsr.data import degradations
from basicsr.utils import img2tensor
from data.base_dataset import BaseDataset


# -------------------------------------------------------------------------
# Multi-PIE configuration
# -------------------------------------------------------------------------

LIGHT_CLASSES = [f"{i:02d}" for i in range(20)]

# 너무 극단적인 측면은 제외한 예시.
# 실제 프로젝트에서 사용 중인 ANGLES_* 배열이 있다면 아래 배열만 교체하면 됨.
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

POSE_CLASSES = ANGLES_MODERATE + ANGLES_NEAR_FRONTAL

# GT/reference로 허용할 포즈.
# 마지막 프레임도 측면일 수 있어야 한다면 POSE_CLASSES와 동일하게 설정해도 됨.
GT_ANGLES = POSE_CLASSES


class MultiFrameMultiPIEDataset(BaseDataset):
    """
    Multi-frame fine-tuning dataset for WaveBFR.

    Directory structure:
        dataroot/
            PID/
                POSE/
                    00.png
                    01.png
                    ...
                    19.png

    Output:
        HR:
            [3, H, W]
            마지막 reference 프레임의 원본 영상.

        LR:
            [max_frames, 3, H, W]
            시간 순서대로 해상도가 증가하는 다중 프레임 입력.
            사용하지 않는 위치는 zero tensor.

        LR_mask:
            [max_frames]
            실제 프레임이면 True, padding이면 False.

        num_frames:
            실제 입력 프레임 개수.

        frame_sizes:
            각 프레임을 degradation할 때 사용한 내부 LR 해상도.
            padding 위치는 0.

        HR_paths:
            GT/reference 이미지 경로.
    """
    
    def __init__(self, opt):
        super().__init__(opt)

        self.img_size = opt.load_size
        self.img_dir = opt.dataroot

        self.min_frames = getattr(opt, "min_frames", 5)
        self.max_frames = getattr(opt, "max_frames", 10)

        self.min_lr_size = getattr(opt, "min_lr_size", 16)
        self.max_lr_size = getattr(opt, "max_lr_size", 112)

        if self.min_frames < 2:
            raise ValueError("min_frames must be at least 2.")

        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must be <= max_frames.")

        if self.min_lr_size > self.max_lr_size:
            raise ValueError("min_lr_size must be <= max_lr_size.")

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Degradation settings
        self.blur_kernel_size = opt.blur_kernel_size
        self.kernel_list = opt.kernel_list
        self.kernel_prob = opt.kernel_prob
        self.blur_sigma = opt.blur_sigma
        self.noise_range = opt.noise_range
        self.jpeg_range = opt.jpeg_range

        # 필요하면 보조 프레임에 reference와 동일한 이미지를 허용할지 결정.
        self.allow_duplicate_frames = getattr(
            opt, "allow_duplicate_frames", False
        )

        # 데이터 목록
        self.samples = []
        self.reference_samples = []
        self.samples_by_pid = defaultdict(list)

        self._build_sample_index()

        if len(self.reference_samples) == 0:
            raise RuntimeError(
                f"No valid Multi-PIE samples were found in: {self.img_dir}"
            )

        print(
            f"[MultiFrameMultiPIEDataset] "
            f"{len(self.samples)} total images, "
            f"{len(self.reference_samples)} reference images, "
            f"{len(self.samples_by_pid)} identities"
        )

    # ---------------------------------------------------------------------
    # Index construction
    # ---------------------------------------------------------------------

    def _build_sample_index(self):
        """
        모든 유효 이미지를 검색하고 PID별로 묶는다.

        sample:
            {
                "pid": str,
                "pose": str,
                "light": str,
                "path": str
            }
        """

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.img_dir}"
            )

        for pid in sorted(os.listdir(self.img_dir)):
            pid_dir = os.path.join(self.img_dir, pid)

            if not os.path.isdir(pid_dir):
                continue

            for pose in POSE_CLASSES:
                pose_dir = os.path.join(pid_dir, pose)

                if not os.path.isdir(pose_dir):
                    continue

                for light in LIGHT_CLASSES:
                    image_path = os.path.join(pose_dir, f"{light}.png")

                    if not os.path.isfile(image_path):
                        continue

                    sample = {
                        "pid": pid,
                        "pose": pose,
                        "light": light,
                        "path": image_path,
                    }

                    self.samples.append(sample)
                    self.samples_by_pid[pid].append(sample)

                    if pose in GT_ANGLES:
                        self.reference_samples.append(sample)

    # ---------------------------------------------------------------------
    # Image loading
    # ---------------------------------------------------------------------

    @staticmethod
    def _random_interpolation():
        return random.choice(
            [
                cv2.INTER_AREA,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
            ]
        )

    def _read_image(self, image_path):
        """
        이미지를 BGR float32 [0, 1] 형태로 읽는다.
        degradation 전에 충분한 해상도를 확보하기 위해 512x512로 resize한다.
        """

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(
                f"Failed to read image: {image_path}"
            )

        interpolation = self._random_interpolation()

        image = cv2.resize(
            image,
            (512, 512),
            interpolation=interpolation,
        )

        return image.astype(np.float32) / 255.0

    # ---------------------------------------------------------------------
    # Frame sampling
    # ---------------------------------------------------------------------

    def _sample_auxiliary_frames(self, reference_sample, num_auxiliary):
        """
        reference와 동일 인물의 다른 포즈/조명 이미지를 무작위로 선택한다.

        마지막 프레임은 reference_sample로 별도 추가되므로,
        여기서는 앞쪽 auxiliary frame들만 반환한다.
        """

        pid = reference_sample["pid"]
        candidates = self.samples_by_pid[pid]

        if not self.allow_duplicate_frames:
            candidates = [
                sample
                for sample in candidates
                if sample["path"] != reference_sample["path"]
            ]

        if len(candidates) == 0:
            # 해당 인물의 이미지가 reference 한 장뿐인 극단적인 경우
            candidates = [reference_sample]

        if len(candidates) >= num_auxiliary:
            auxiliary = random.sample(candidates, num_auxiliary)
        else:
            # 이미지 개수가 부족한 PID는 중복 허용
            auxiliary = random.choices(
                candidates,
                k=num_auxiliary,
            )

        return auxiliary

    def _sample_lr_sizes(self, num_frames):
        """
        16~112 사이를 num_frames개의 순차적인 구간으로 나누고,
        각 구간에서 하나의 LR 크기를 선택한다.

        예: num_frames=5
            frame 1: 약 16~35
            frame 2: 약 35~54
            frame 3: 약 54~74
            frame 4: 약 74~93
            frame 5: 약 93~112

        반환되는 크기는 항상 비감소 순서이다.
        """

        # 정수 구간 경계.
        edges = np.linspace(
            self.min_lr_size,
            self.max_lr_size,
            num_frames + 1,
        )

        lr_sizes = []

        for frame_idx in range(num_frames):
            lower = int(round(edges[frame_idx]))
            upper = int(round(edges[frame_idx + 1]))

            lower = max(self.min_lr_size, lower)
            upper = min(self.max_lr_size, upper)

            if upper < lower:
                upper = lower

            lr_size = random.randint(lower, upper)
            lr_sizes.append(lr_size)

        # 구간 경계의 반올림으로 역전되는 경우를 방지.
        lr_sizes = sorted(lr_sizes)

        return lr_sizes

    # ---------------------------------------------------------------------
    # Degradation
    # ---------------------------------------------------------------------

    def _generate_lq(self, hr_image, lr_size):
        """
        HR 이미지 한 장을 지정된 내부 해상도 lr_size로 열화한 뒤,
        최종적으로 img_size x img_size로 다시 확대한다.

        예:
            512x512
              -> blur
              -> 24x24
              -> noise / JPEG
              -> 112x112
        """

        interpolation = self._random_interpolation()

        # Avoid modifying the original NumPy array.
        lq_image = hr_image.copy()

        # 1. Blur
        if self.blur_kernel_size is not None:
            min_kernel, max_kernel = self.blur_kernel_size

            if min_kernel >= max_kernel:
                raise ValueError(
                    "blur_kernel_size[0] must be smaller than "
                    "blur_kernel_size[1]."
                )

            kernel_size = random.randint(min_kernel, max_kernel) * 2 + 1

            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None,
            )

            lq_image = cv2.filter2D(
                lq_image,
                -1,
                kernel,
            )

        # 2. Downsample to the assigned frame resolution
        lr_size = int(np.clip(
            lr_size,
            self.min_lr_size,
            self.max_lr_size,
        ))

        lq_image = cv2.resize(
            lq_image,
            (lr_size, lr_size),
            interpolation=interpolation,
        )

        # 3. Noise
        if self.noise_range is not None:
            lq_image = degradations.random_add_gaussian_noise(
                lq_image,
                self.noise_range,
            )

        # 4. JPEG compression
        if self.jpeg_range is not None:
            lq_image = degradations.random_add_jpg_compression(
                lq_image,
                self.jpeg_range,
            )

        # 5. Resize to the network input resolution
        lq_image = cv2.resize(
            lq_image,
            (self.img_size, self.img_size),
            interpolation=interpolation,
        )

        lq_image = np.clip(lq_image, 0.0, 1.0)

        return lq_image

    def _to_normalized_tensor(self, image):
        """
        BGR HWC NumPy [0,1]
            -> RGB CHW tensor
            -> [-1,1] normalization
        """

        tensor = img2tensor(
            image,
            bgr2rgb=True,
            float32=True,
        )

        normalize(
            tensor,
            self.mean,
            self.std,
            inplace=True,
        )

        return tensor

    def _to_lq_tensor(self, image):
        """
        기존 WaveBFR 코드와 동일하게 8-bit rounding 후 normalize한다.
        """

        tensor = img2tensor(
            image,
            bgr2rgb=True,
            float32=True,
        )

        tensor = torch.clamp(
            (tensor * 255.0).round(),
            0,
            255,
        ) / 255.0

        normalize(
            tensor,
            self.mean,
            self.std,
            inplace=True,
        )

        return tensor

    # ---------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------

    def __getitem__(self, index):
        reference_sample = self.reference_samples[index]

        # 실제 사용할 프레임 개수: 5~10
        num_frames = random.randint(
            self.min_frames,
            self.max_frames,
        )

        # 앞쪽 T-1개는 동일 PID의 랜덤 포즈/조명
        auxiliary_samples = self._sample_auxiliary_frames(
            reference_sample=reference_sample,
            num_auxiliary=num_frames - 1,
        )

        # 마지막 프레임은 반드시 reference/GT와 동일한 원본
        sequence_samples = auxiliary_samples + [reference_sample]

        # 시간 순서가 뒤로 갈수록 해상도가 증가
        lr_sizes = self._sample_lr_sizes(num_frames)

        # 마지막 프레임의 원본이 학습 GT
        reference_hr_numpy = self._read_image(
            reference_sample["path"]
        )

        gt_resized = cv2.resize(
            reference_hr_numpy,
            (self.img_size, self.img_size),
            interpolation=self._random_interpolation(),
        )

        hr_tensor = self._to_normalized_tensor(gt_resized)

        # 고정 크기 다중 프레임 tensor
        lr_tensor = torch.zeros(
            self.max_frames,
            3,
            self.img_size,
            self.img_size,
            dtype=torch.float32,
        )

        lr_mask = torch.zeros(
            self.max_frames,
            dtype=torch.bool,
        )

        frame_sizes = torch.zeros(
            self.max_frames,
            dtype=torch.long,
        )

        frame_paths = [""] * self.max_frames
        frame_poses = [""] * self.max_frames
        frame_lights = [""] * self.max_frames

        for frame_idx, (sample, lr_size) in enumerate(
            zip(sequence_samples, lr_sizes)
        ):
            frame_hr = self._read_image(sample["path"])

            frame_lq = self._generate_lq(
                hr_image=frame_hr,
                lr_size=lr_size,
            )

            lr_tensor[frame_idx] = self._to_lq_tensor(frame_lq)
            lr_mask[frame_idx] = True
            frame_sizes[frame_idx] = lr_size

            frame_paths[frame_idx] = sample["path"]
            frame_poses[frame_idx] = sample["pose"]
            frame_lights[frame_idx] = sample["light"]

        return {
            "HR": hr_tensor,
            "LR": lr_tensor,
            "LR_mask": lr_mask,
            "num_frames": torch.tensor(
                num_frames,
                dtype=torch.long,
            ),
            "frame_sizes": frame_sizes,
            "HR_paths": reference_sample["path"],
            "LR_paths": frame_paths,
            "poses": frame_poses,
            "lights": frame_lights,
            "pid": reference_sample["pid"],
        }

    def __len__(self):
        return len(self.reference_samples)
    
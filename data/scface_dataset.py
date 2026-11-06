import os
from glob import glob
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor

from data.base_dataset import BaseDataset


class SCFaceDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.img_dir = opt.dataroot

        self.gallery_paths = []
        self.lq_paths = []

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        print("# Loading image paths...")

        filenames = sorted(os.listdir(os.path.join(self.img_dir, "lq")))

        for filename in filenames:
            pid = filename.split("_")[0]
            gallery_path = os.path.join(self.img_dir, "gallery", "%s_frontal.png" % pid)
            lq_path = os.path.join(self.img_dir, "lq", filename)

            self.gallery_paths.append(gallery_path)
            self.lq_paths.append(lq_path)

        print("# The number of LQ images:", len(self.lq_paths))

    def __getitem__(self, index):
        gallery_img = cv2.imread(self.gallery_paths[index])
        lq_img = cv2.imread(self.lq_paths[index])

        h, w = gallery_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            gallery_img = cv2.resize(
                gallery_img,
                dsize=(self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )

        h, w = lq_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            lq_img = cv2.resize(
                lq_img,
                dsize=(self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )

        gallery_img = gallery_img.astype(np.float32) / 255.0
        lq_img = lq_img.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW, numpy to tensor
        gallery_img, lq_img = img2tensor(
            [gallery_img, lq_img], bgr2rgb=True, float32=True
        )

        # normalize
        normalize(gallery_img, self.mean, self.std, inplace=True)
        normalize(lq_img, self.mean, self.std, inplace=True)

        return {"HR": gallery_img, "LR": lq_img, "HR_paths": self.lq_paths[index]}

    def __len__(self):
        return len(self.lq_paths)

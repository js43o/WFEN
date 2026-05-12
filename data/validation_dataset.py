import os
from glob import glob
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor

from data.base_dataset import BaseDataset


class ValidationDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.img_dir = opt.dataroot
        
        self.gt_paths = []
        self.lq_paths = []
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
        print("# Loading image paths...")

        filenames = sorted(os.listdir(os.path.join(self.img_dir, 'lq')))
        
        for filename in filenames:
            gt_path = os.path.join(self.img_dir, 'gt', filename)
            lq_path = os.path.join(self.img_dir, 'lq', filename)
            
            self.gt_paths.append(gt_path)
            self.lq_paths.append(lq_path)
        
        assert len(self.gt_paths) == len(self.lq_paths), "The number of GT and LQ images should be the same."
        
        print("# The number of images:", len(self.gt_paths))

    def __getitem__(self, index):
        # load gt image
        gt_img = cv2.imread(self.gt_paths[index])
        lq_img = cv2.imread(self.lq_paths[index])
        
        h, w = gt_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            gt_img = cv2.resize(
                gt_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
        
        h, w = lq_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            lq_img = cv2.resize(
                lq_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
        
        gt_img = gt_img.astype(np.float32) / 255.0
        lq_img = lq_img.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW, numpy to tensor
        gt_img, lq_img = img2tensor([gt_img, lq_img], bgr2rgb=True, float32=True)

        # normalize
        normalize(gt_img, self.mean, self.std, inplace=True)
        normalize(lq_img, self.mean, self.std, inplace=True)

        return {'HR': gt_img, 'LR': lq_img, 'HR_paths': self.gt_paths[index]}
    
    def __len__(self,):
        return len(self.gt_paths)

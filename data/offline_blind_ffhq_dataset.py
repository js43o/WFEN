import os
from glob import glob
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor

from data.base_dataset import BaseDataset


class OfflineBlindFFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.img_dir = opt.dataroot
        
        self.hr_paths = []
        self.lr_paths = []
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
        epochs = sorted(os.listdir(os.path.join(self.img_dir, 'lq')))
        
        print("# Loading image paths...")
        
        for epoch in epochs:
            hr_paths_epoch = sorted(glob(os.path.join(self.img_dir, 'gt', '*.jpg')))
            lr_paths_epoch = sorted(glob(os.path.join(self.img_dir, 'lq', epoch, '*.jpg')))
            
            assert len(hr_paths_epoch) == len(lr_paths_epoch), "The number of HR and LR images should be the same."
            
            self.hr_paths.extend(hr_paths_epoch)
            self.lr_paths.extend(lr_paths_epoch)

        print("# The number of images:", len(self.hr_paths))

    def __getitem__(self, index):
        # load gt image
        hr_img = cv2.imread(self.hr_paths[index])
        lr_img = cv2.imread(self.lr_paths[index])
        
        h, w = hr_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            hr_img = cv2.resize(
                hr_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
        
        h, w = lr_img.shape[:2]
        if h != self.img_size or w != self.img_size:
            lr_img = cv2.resize(
                lr_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
        
        hr_img = hr_img.astype(np.float32) / 255.0
        lr_img = lr_img.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW, numpy to tensor
        hr_img, lr_img = img2tensor([hr_img, lr_img], bgr2rgb=True, float32=True)

        # normalize
        normalize(hr_img, self.mean, self.std, inplace=True)
        normalize(lr_img, self.mean, self.std, inplace=True)

        return {'HR': hr_img, 'LR': lr_img, 'HR_paths': self.hr_paths[index]}
    
    def __len__(self,):
        return len(self.hr_paths)

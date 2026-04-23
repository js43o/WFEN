import os
from glob import glob
import random
import numpy as np
import cv2
import math
import torch
from torchvision.transforms.functional import normalize
from basicsr.data import degradations as degradations
from basicsr.utils import img2tensor

from data.base_dataset import BaseDataset


class CleanFFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.shuffle = True if opt.isTrain else False 

        self.img_dir = opt.dataroot
        self.img_names = self.get_img_names()

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def get_img_names(self,):
        img_names = []
        for ext in ['png', 'jpg', 'jpeg']:
            img_names.extend([x for x in glob(os.path.join(self.img_dir, '*.' + ext))])
            img_names.extend([x for x in glob(os.path.join(self.img_dir, '**/*.' + ext))])
            
        img_names.sort()
        
        if self.shuffle:
            random.shuffle(img_names)
            
        print("# The number of images:", len(img_names))
            
        return img_names

    def __getitem__(self, index):
        # load gt image
        img_path = os.path.join(self.img_dir, self.img_names[index])
        hr_img = cv2.imread(img_path)
        hr_img = cv2.resize(
            hr_img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR
        )   # resize for degradation
        hr_img = hr_img.astype(np.float32) / 255.0

        hr_img = cv2.resize(hr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        hr_img = img2tensor(hr_img, bgr2rgb=True, float32=True)

        # normalize
        normalize(hr_img, self.mean, self.std, inplace=True)

        return {'HR': hr_img, 'LR': hr_img, 'HR_paths': img_path}
    
    def __len__(self,):
        return len(self.img_names)

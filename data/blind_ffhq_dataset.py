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


class BlindFFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.shuffle = True if opt.isTrain else False 

        self.img_dir = opt.dataroot
        self.img_names = self.get_img_names()

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # degradations
        self.blur_kernel_size = opt.blur_kernel_size
        self.kernel_list = opt.kernel_list
        self.kernel_prob = opt.kernel_prob
        self.blur_sigma = opt.blur_sigma
        self.downsample_range = opt.downsample_range
        self.noise_range = opt.noise_range
        self.jpeg_range = opt.jpeg_range
        

    def get_img_names(self,):
        img_names = [x for x in glob(os.path.join(self.img_dir, '*/**.png'))] 
        if self.shuffle:
            random.shuffle(img_names)
        return img_names

    def __getitem__(self, index):
        # load gt image
        img_path = os.path.join(self.img_dir, self.img_names[index])
        hr_img = cv2.imread(img_path)
        hr_img = cv2.resize(
            hr_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        hr_img = hr_img.astype(np.float32) / 255.0

        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0],self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        lr_img = cv2.filter2D(hr_img, -1, kernel)
        
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        lr_img = cv2.resize(lr_img, (int(self.img_size // scale), int(self.img_size // scale)), interpolation=cv2.INTER_LINEAR)
        
        # noise
        if self.noise_range is not None:
            lr_img = degradations.random_add_gaussian_noise(lr_img, self.noise_range)
            
        # jpeg compression
        if self.jpeg_range is not None:
            lr_img = degradations.random_add_jpg_compression(lr_img, self.jpeg_range)

        # resize to original size
        lr_img = cv2.resize(lr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        hr_img, lr_img = img2tensor([hr_img, lr_img], bgr2rgb=True, float32=True)

        # round and clip
        lr_img = torch.clamp((lr_img * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(hr_img, self.mean, self.std, inplace=True)
        normalize(lr_img, self.mean, self.std, inplace=True)

        return {'HR': hr_img, 'LR': lr_img, 'HR_paths': img_path}
    
    def __len__(self,):
        return len(self.img_names)

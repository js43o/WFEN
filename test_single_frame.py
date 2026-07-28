import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    if len(opt.pretrain_model_path):
        model.load_pretrain_model()
    else:
        model.setup(opt)  # regular setup: load and print networks; create schedulers

    if len(opt.save_as_dir):
        save_dir = opt.save_as_dir
    else:
        save_dir = os.path.join(
            opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
        )
        if opt.load_iter > 0:  # load_iter is 0 by default
            save_dir = "{:s}_iter{:d}".format(save_dir, opt.load_iter)
    os.makedirs(save_dir, exist_ok=True)

    print("creating result directory", save_dir)

    # os.makedirs(os.path.join(save_dir, 'lr'), exist_ok=True)
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'hr'), exist_ok=True)

    network = model.netG
    network.eval()
    
    sr_save_dir = save_dir
    # lr_save_dir = os.path.join(save_dir, "LR")

    os.makedirs(sr_save_dir, exist_ok=True)
    # os.makedirs(lr_save_dir, exist_ok=True)

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        multi_frame_lr = data["LR"].to(opt.data_device)
        frame_mask = data["LR_mask"].to(opt.data_device).bool()

        # multi_frame_lr: [B, T, C, H, W]
        # frame_mask:     [B, T]
        batch_size = multi_frame_lr.size(0)

        # 각 샘플의 마지막 유효 프레임 위치
        last_frame_indices = frame_mask.long().sum(dim=1) - 1

        if torch.any(last_frame_indices < 0):
            raise RuntimeError("LR_mask에 유효한 프레임이 없는 샘플이 있습니다.")

        batch_indices = torch.arange(
            batch_size,
            device=multi_frame_lr.device,
        )

        # [B, C, H, W]
        reference_lr = multi_frame_lr[
            batch_indices,
            last_frame_indices,
        ]

        with torch.no_grad():
            output = network(reference_lr)

            if isinstance(output, tuple):
                output = output[0]

        img_paths = data["HR_paths"]

        for batch_idx in range(batch_size):
            sr_img = utils.tensor_to_img(
                output[batch_idx],
                normal=True,
            )
            # lr_img = utils.tensor_to_img(
            #     reference_lr[batch_idx],
            #     normal=True,
            # )

            img_path = img_paths[batch_idx]

            if opt.dataset_name == "multi_frame_multipie":
                filename = "_".join(
                    os.path.normpath(img_path).split(os.sep)[-3:]
                )
            elif opt.dataset_name == "multi_frame_kface":
                filename = "_".join(
                    os.path.normpath(img_path).split(os.sep)[-5:]
                )
            else:
                filename = os.path.basename(img_path)

            Image.fromarray(sr_img).save(
                os.path.join(sr_save_dir, filename)
            )
            # Image.fromarray(lr_img).save(
            #     os.path.join(lr_save_dir, filename)
            # )

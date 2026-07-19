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
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    
    device = 'cuda'
    
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

    for filename in tqdm(sorted(os.listdir(opt.test_img_path))):
        img_path = os.path.join(opt.test_img_path, filename)
        img = utils.img_to_tensor(img_path, device=device)

        with torch.no_grad():
            output = network(img)
            if isinstance(output, tuple):
                output = output[0]  # except commit_loss

        sr_img = utils.tensor_to_img(output, normal=True)
        Image.fromarray(sr_img).save(os.path.join(save_dir, filename))

import torch
import torch.nn as nn
import torch.optim as optim

from models import loss
from models import networks
from .base_model import BaseModel
from utils import utils
from models.arch.wfen import VQWFEN


class VQWFENModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--lambda_pix', type=float, default=100.0, help='weight for pixel loss')
            parser.add_argument('--lambda_pcp', type=float, default=1.0, help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_commit', type=float, default=1000.0, help='weight for commitment loss')
            
        parser.add_argument(
            "--scale_factor", type=int, default=8, help="upscale factor for model"
        )
        
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = VQWFEN(is_pretrain=opt.is_pretrain)
        self.netG = networks.define_network(opt, self.netG)

        self.model_names = ["G"]
        self.load_model_names = ["G"]
        self.loss_names = ["Pix", "PCP", "Commit"]
        self.visual_names = ["img_LR", "img_SR", "img_HR"]

        if self.isTrain:
            self.vgg19 = loss.PCPFeat('./pretrain_models/vgg19-dcbb9e9d.pth', 'vgg')
            self.vgg19 = networks.define_network(opt, self.vgg19, isTrain=False, init_network=False)

            self.criterionL1 = nn.L1Loss()
            self.criterionPCP = loss.PCPLoss(opt)
            
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99)
            )
            self.optimizers = [self.optimizer_G]
        
        if not opt.is_pretrain:
            for name, param in self.netG.named_parameters():
                if name.startswith("module.Transformer.") or name.startswith("module.TransformerUp"):
                    print("❄️ Freeze layer:", name)
                    param.requires_grad = False # freeze all layers after VQ
            

    def load_pretrain_model(
        self,
    ):
        print("Loading pretrained model", self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.module.load_state_dict(weight)

    def for_load_pretrain_model(self, path):
        print("Loading pretrained model", path)
        weight = torch.load(path)
        self.netG.module.load_state_dict(weight)

    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input["LR"].to(self.opt.data_device)
        self.img_HR = input["HR"].to(self.opt.data_device)

    def forward(self):
        self.img_SR, self.loss_Commit = self.netG(self.img_LR)
        self.loss_Commit = self.loss_Commit * self.opt.lambda_commit

        self.fake_vgg_feat = self.vgg19(self.img_SR)
        self.real_vgg_feat = self.vgg19(self.img_HR)

    def backward_G(self):
        # Pix loss
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        self.loss_PCP = self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat) * self.opt.lambda_pcp 
        
        loss = self.loss_Pix + self.loss_PCP + self.loss_Commit
        
        loss.backward()

    def optimize_parameters(
        self,
    ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self, size=128):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        return visual_imgs

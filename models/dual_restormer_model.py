import torch
import torch.nn as nn
import torch.optim as optim
import pyiqa

from models import loss, networks
from .base_model import BaseModel
from utils import utils
from models.arch.restormer import Restormer
from models.arch.dual_wavelet_restormer import WaveletRestormer

from models.arch.wfen import HaarWavelet
from helpers.arcface.models import resnet_face18


class DualRestormerModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        parser.add_argument('--scale_factor', type=int, default=8, help='upscale factor for model')
        parser.add_argument('--lambda_pix', type=float, default=0.0, help='weight for pixel loss')
        parser.add_argument('--lambda_ssim', type=float, default=0.0, help='weight for SSIM loss')
        parser.add_argument('--lambda_vgg', type=float, default=0.0, help='weight for VGG loss')
        parser.add_argument('--lambda_adv', type=float, default=0.001, help='weight for adversarial loss')
        parser.add_argument('--lambda_id', type=float, default=0.0, help='weight for identity loss')
        
        parser.add_argument('--lambda_lf', type=float, default=1.0, help='weight for low-frequency (LF) feature loss')
        parser.add_argument('--lambda_hf', type=float, default=1.0, help='weight for high-frequency (HF) feature loss')
        return parser
    

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.in_channels = 3

        self.netG = WaveletRestormer()
        self.netG = networks.define_network(opt, self.netG)
        self.wavelet_transform = HaarWavelet(in_channels=self.in_channels, grad=False).to(device=opt.data_device)

        self.model_names = ['G']
        self.load_model_names = ['G']
        self.loss_names = ['Pix', 'LF', 'HF'] 
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G]
            
            if opt.lambda_ssim > 0:
                print("➕ SSIM loss")
                self.loss_names.append('SSIM')
                self.compute_ssim = pyiqa.create_metric("ssim", as_loss=True, device=self.opt.data_device)

            if opt.lambda_vgg > 0:
                print("➕ VGG loss")
                self.loss_names.append('VGG')
                self.criterionPCP = loss.PCPLoss(opt)
                
                self.vgg19 = loss.PCPFeat('./pretrain_models/vgg19-dcbb9e9d.pth', 'vgg')
                self.vgg19 = networks.define_network(opt, self.vgg19, isTrain=False, init_network=False)
            
            if opt.lambda_adv > 0:
                print("➕ Adv loss")
                self.model_names.append('D')
                self.load_model_names.append('D')
                self.loss_names.extend(['FM', 'G', 'D'])
                
                self.criterionFM = loss.FMLoss().to(opt.data_device)
                self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)
                
                self.netD = networks.MultiScaleDiscriminator(3, n_layers=opt.n_layers_D, norm_type=opt.Dnorm, num_D=opt.num_D)
                self.netD = networks.define_network(opt, self.netD, use_norm='spectral_norm')
                
                self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.99))
                self.optimizers.append(self.optimizer_D)

            if opt.lambda_id > 0:
                print("➕ identity loss")
                self.loss_names.append('ID')
                self.criterionID = nn.CosineEmbeddingLoss()
                self.arcface_model = resnet_face18(use_se=False).to(opt.data_device)
                self.arcface_model.load_state_dict(
                    torch.load("helpers/arcface/weights/resnet18_110_wo_dist.pth", weights_only=False)
                )
                self.arcface_model.requires_grad_(False)
                self.arcface_model.eval()
                
                
    def load_pretrain_model(self,):
        print('Loading pretrained model', self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.module.load_state_dict(weight)

    def for_load_pretrain_model(self, path):
        print('Loading pretrained model', path)
        weight = torch.load(path)
        self.netG.module.load_state_dict(weight)
    
    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(self.opt.data_device)
        self.img_HR = input['HR'].to(self.opt.data_device)
        
        haar = self.wavelet_transform(self.img_HR, rev=False)
        self.img_lf_HR = haar.narrow(1, 0, self.in_channels).to(self.opt.data_device)
        h = haar.narrow(1, self.in_channels, self.in_channels)
        v = haar.narrow(1, self.in_channels * 2, self.in_channels)
        d = haar.narrow(1, self.in_channels * 3, self.in_channels)
        self.img_hf_HR = torch.cat([h, v, d], 1).to(self.opt.data_device)

    def forward(self):
        self.img_SR, self.img_lf_SR, self.img_hf_SR = self.netG(self.img_LR)
        
        if self.opt.lambda_vgg > 0:
            self.fake_vgg_feat = self.vgg19(self.img_lf_SR)
            self.real_vgg_feat = self.vgg19(self.img_lf_HR)
        
        if self.opt.lambda_adv > 0:
            self.real_D_results = self.netD(self.img_HR, return_feat=True)
            self.fake_D_results = self.netD(self.img_SR.detach(), return_feat=False)
            self.fake_G_results = self.netD(self.img_SR, return_feat=True)


    def backward_G(self):
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        self.loss_LF = self.criterionL1(self.img_lf_SR, self.img_lf_HR) * self.opt.lambda_lf
        self.loss_HF = self.criterionL1(self.img_hf_SR, self.img_hf_HR) * self.opt.lambda_hf
        
        loss = self.loss_Pix + self.loss_LF + self.loss_HF
        
        if self.opt.lambda_ssim > 0:
            self.loss_SSIM = (1 - self.compute_ssim(self.img_SR, self.img_HR)) * self.opt.lambda_ssim
            loss += self.loss_SSIM
        
        if self.opt.lambda_vgg > 0:
            self.loss_VGG = self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat) * self.opt.lambda_vgg
            loss += self.loss_VGG
        
        if self.opt.lambda_adv > 0:
            # Feature matching loss
            tmp_loss =  0
            for i in range(self.opt.num_D):
                tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) 
            self.loss_FM = tmp_loss * (self.opt.lambda_adv * 10) / self.opt.num_D   # generator loss보다 10배 높게 할당

            # Generator loss
            tmp_loss = 0
            for i in range(self.opt.num_D):
                tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
            self.loss_G = tmp_loss * self.opt.lambda_adv / self.opt.num_D
            
            loss += self.loss_FM + self.loss_G
        
        if self.opt.lambda_id > 0:
            pred_embed = self.arcface_model(utils.process_arcface_input(self.img_lf_SR))
            hr_embed = self.arcface_model(utils.process_arcface_input(self.img_lf_HR))
            ID_TARGET = torch.ones((hr_embed.shape[0],), device=hr_embed.device)

            self.loss_ID = self.criterionID(pred_embed, hr_embed, ID_TARGET) * self.opt.lambda_id
            loss += self.loss_ID
        
        loss.backward()
    
    def backward_D(self):
        loss = 0
        for i in range(self.opt.num_D):
            loss += 0.5 * (self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0], True))
        self.loss_D = loss / self.opt.num_D 
        self.loss_D.backward()
        
    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        if self.opt.lambda_adv > 0:
            # ---- Update D ------------
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
 
    def get_current_visuals(self, size=128):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]
        
        return visual_imgs

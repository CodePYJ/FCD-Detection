from generative.networks.nets import VQVAE
import torch.nn as nn
import torch
from kornia.losses import SSIMLoss
import os
from PIL import Image
import numpy as np

class LdmVQVAE(VQVAE):
    def __init__(self, log_image_epoch, log_path, ckpt_path=None, *args, **kwargs):
        super().__init__(num_channels=(128, 256),downsample_parameters=((2, 4, 1, 1),(2, 4, 1, 1)), upsample_parameters=((2, 4, 1, 1, 0),(2, 4, 1, 1, 0)), *args, **kwargs)
        if ckpt_path is not None:
            check_point = torch.load(ckpt_path)
            self.load_state_dict(check_point["state_dict"])
        self.log_image_epoch = log_image_epoch
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    def get_input(self, batch):
        img, number = batch
        img = img.to(self.device)
        return img, number
    
    def vqvae_loss(self, recon_x, x, quantized_loss,mse_weight=1.0,ssim_weight=1.0,data_range=2.0):
        # 重建损失
        # MSE损失（保持与原代码兼容性）
        mse_loss = nn.MSELoss(reduction='mean')(recon_x, x)

        # SSIM损失（使用kornia实现）
        ssim_loss = SSIMLoss(
            window_size=11,
            reduction='mean',
            max_val=data_range  # 关键参数！根据输入数据范围调整
        )(recon_x, x)

        # 组合多尺度重建损失
        reconstruction_loss = (
                mse_weight * mse_loss +
                ssim_weight * ssim_loss
        )

        return reconstruction_loss + quantized_loss
    
    def configure_optimizers(self, lr):
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return [opt]
    
    def log_images(self, epoch, imgs, latent, N=2):
        if N>imgs.shape[0]:
            N = imgs.shape[0]
        for i in range(N):
            for ch in range(imgs.shape[1]):
                img = imgs[i][ch]
                img = (img - img.min())/(img.max()-img.min())*255.0
                img = img.to(torch.uint8).cpu().numpy()
                Image.fromarray(img).convert('L').save(os.path.join(self.log_path, str(epoch)+f'_{str(ch)}_{str(i)}'+".png"))
            for ch in range(latent.shape[1]):
                latent_img = latent[i][ch]
                latent_img = (latent_img - latent_img.min())/(latent_img.max()-latent_img.min())*255.0
                latent_img = latent_img.to(torch.uint8).cpu().numpy()
                Image.fromarray(latent_img).convert('L').save(os.path.join(self.log_path, str(epoch)+f'_{str(ch)}_{str(i)}'+"_latent.png"))
                
    def end_epoch(self, epoch, batch):
        img, number = self.get_input(batch)
        recon_batch, quantized_loss = self(img)
        latent = self.encode_stage_2_inputs(img)
        if (epoch+1) % self.log_image_epoch == 0:
            self.log_images(epoch, img, latent)
    
    def training_step(self, batch, optim_idx, epoch, device):
        self.device = device
        img, number = self.get_input(batch)
        recon_batch, quantized_loss = self(img)
        loss = self.vqvae_loss(recon_batch, img, quantized_loss)
        return loss
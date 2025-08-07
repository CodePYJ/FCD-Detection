from models.diffusion.ddpm2 import DDPM
from modules.utils import *

class LDM(DDPM):
    def __init__(self, vqvae_config=None, condition_type="none", *args, **kwargs):
        super().__init__(condition_type=condition_type, *args, **kwargs)
        self.vqvae = instantiate_from_config(vqvae_config)
        self.vqvae.eval()
        self.condition_type = condition_type
        self.mean_flair : float = 1.2603
        self.mean_t1 : float = 1.0541
        self.std_flair : float = 3.1313
        self.std_t1 : float = 3.1982
        self.device = None
    
    def encode(self, x_start, cond_img):
        z = (self.vqvae.encode_stage_2_inputs(x_start) - self.mean_flair) / self.std_flair# 获取量化后的潜变量
        if(self.condition_type == "flair"):
            cond_z = (self.vqvae.encode_stage_2_inputs(cond_img) - self.mean_flair) /self.std_flair
        elif(self.condition_type == "t1"):
            cond_z = (self.vqvae.encode_stage_2_inputs(cond_img) - self.mean_t1) /self.std_t1
            
        return z, cond_z
    
    def decode(self, xt):
        xt = xt * self.std_flair +self.mean_flair
        xt = self.vqvae.decode_stage_2_outputs(xt)
        return xt
    
    def get_input(self, batch):
        x_start, cond_img, is_fcd, seg, number, context_mask = super().get_input(batch)
        z, cond_z = self.encode(x_start, cond_img)
        z = z.to(self.device)
        cond_z = cond_z.to(self.device)
        return z, cond_z, x_start, is_fcd, seg, number, context_mask

    def training_step(self, batch, optim_idx, epoch, device):
        if self.device is None:
            self.device = device
            self.vqvae.to(self.device)
        x_start, cond_img, _, is_fcd, seg, number, context_mask = self.get_input(batch)
        loss = self(x_start, cond_img, is_fcd, context_mask)
        return loss
    
    def testing_step(self, batch, sample_path, device, noise_level, th):
        self.device = device
        self.sample_path = sample_path
        self.noise_scheduler.set_timesteps(noise_level, device=device)
        
        z, cond_z, _, is_fcd, seg, number, context_mask = self.get_input(batch)
        
        if self.condition_type == "none":
            sample_imgs = self.sample(z.shape[0], None)
        else:
            sample_imgs = self.sample(z.shape[0], z, cond_z)
        
        sample_imgs = sample_imgs*self.std_flair +self.mean_flair
        sample_imgs = self.vqvae.decode_stage_2_outputs(sample_imgs)
        batch_mean_dice, dice_dict = self.metric(sample_imgs, z, seg, number, th)
        return batch_mean_dice, dice_dict
        
    def end_epoch(self, epoch, batch):
        if (epoch+1) % self.log_image_epoch == 0:
            z, cond_z, _, is_fcd, seg, number, context_mask = self.get_input(batch)
            self.log_images(epoch, z, cond_z, label=is_fcd)
import torch
import os
from models.unet.openaiunet import EncoderUNetModel
import blobfile as bf
import io
from diffusers import DDPMScheduler
import torch.nn.functional as F

class Classifier(EncoderUNetModel):
    def __init__(self, linear_start, linear_end, timesteps, ckpt_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_timesteps = timesteps
        
        if ckpt_path is not None:
            check_point = torch.load(ckpt_path)
            self.load_state_dict(check_point["state_dict"])
            # self.load_state_dict(check_point)
            
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=linear_start,
            beta_end=linear_end,
            beta_schedule="linear"
        )
            
    def get_input(self, batch):
        img, seg, is_fcd, number = batch
        img = img.to(self.device)
        is_fcd = is_fcd.to(self.device)
        return img, is_fcd
    
    def configure_optimizers(self, lr):
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return [opt]
    
    def training_step(self, batch, optim_idx, epoch, device):
        self.device = device
        x_start, is_fcd = self.get_input(batch)
        
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
        
        logits = self(x_noisy, t)
        lobe_weight = None
        if logits.shape[1] > 2:
            lobe_weight = torch.tensor([0.5, 0.988, 0.911, 0.682, 0.988, 0.994, 0.982, 0.982, 0.97]).to(device)
            
        loss = F.cross_entropy(logits, is_fcd, reduction="none", weight=lobe_weight).mean()
        
        return loss
    
    def validation_step(self, batch):
        x_start, is_fcd = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
        
        logits = self(x_noisy, t)
        loss = F.cross_entropy(logits, is_fcd, reduction="none").mean()
        return loss
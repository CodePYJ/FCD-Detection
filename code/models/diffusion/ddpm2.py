import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from modules.utils import *
import os
from tqdm import tqdm
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
import cv2

class DDPM(nn.Module):
    def __init__(self, unet_config, classifier_config=None, guidance='none', drop_prob=0.1,use_ddim=False,
                 timesteps=1000, linear_start=1e-4, linear_end=2e-2, 
                 image_size=256, channels=2, condition_type="none", 
                 log_image_epoch=10, log_path="./log", log_noise=200):
        super().__init__()
        self.test_init = 0
        self.guidance = guidance
        self.classifier_config = classifier_config
        self.drop_prob = drop_prob
        self.device = None
        self.sample_path = None
        self.num_timesteps = timesteps
        self.model = instantiate_from_config(unet_config)
        self.log_image_epoch = log_image_epoch
        self.image_size = image_size
        self.channels = channels
        self.log_path = log_path
        self.log_noise = log_noise
        self.use_ddim = use_ddim
        if use_ddim:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=timesteps,
                beta_start=linear_start,
                beta_end=linear_end,
                beta_schedule="linear"
            )
            self.inverse_scheduler = DDIMInverseScheduler(
                num_train_timesteps=timesteps,
                beta_start=linear_start,
                beta_end=linear_end,
                beta_schedule="linear")
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_start=linear_start,
                beta_end=linear_end,
                beta_schedule="linear"
            )
        self.condition_type = condition_type
        self.use_condition = False
        if self.condition_type != "none":
            self.use_condition = True
            
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
    def configure_optimizers(self, lr):
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return [opt]
    
    def cond_fn(self, x, t,  y=None, classifier_scale=1.):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier_model(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=torch.autograd.grad(selected.sum(), x_in)[0]
            return  a * classifier_scale
    
    def sampe_step(self, x_t, cond_img, t, y, context_mask, N=None, sample_type="cg", pre_noise=None):
        t_batch = torch.tensor([t], device=self.device).repeat(N)
        if cond_img is not None:
            model_input = torch.cat((x_t, cond_img), dim=1)
        else:
            model_input = x_t
        if pre_noise is None:
            model_out = self.model(model_input, t_batch, y=y, context_mask=context_mask)
        else:
            model_out = pre_noise
        if sample_type == "cfg":
            eps1 = model_out[N//2:]
            eps2 = model_out[:N//2]
            eps = (1+self.guide_w)*eps1 - self.guide_w*eps2
            # eps = eps2 + self.guide_w*(eps1 - eps2)
            x_t = x_t[:N//2]
            x_t = self.noise_scheduler.step(eps, t, x_t, eta=self.eta, use_clipped_model_output=True).prev_sample
            x_t = x_t.repeat(2,1,1,1)
        elif sample_type == "cg":
            x_t = self.noise_scheduler.step(model_out, t, x_t, eta=self.eta, use_clipped_model_output=True).prev_sample
            x_t = x_t.clamp(-1, 1)
            x_t += self.cond_fn(x_t, t_batch, y, self.guide_w)
        elif sample_type == "inver_ddim":
            x_t = self.inverse_scheduler.step(model_out, t, x_t, False)[0]
        else:
            x_t = self.noise_scheduler.step(model_out, t, x_t, eta=self.eta, use_clipped_model_output=True).prev_sample
            x_t = x_t.clamp(-1, 1)
        return x_t, model_out
    
    def sample(self, N=8, x_start=None, cond_img=None, label=None, context_mask=None, sample_timesteps=None):
        image_size = self.image_size
        channels = self.channels
        
        if x_start is None:
            x_t = torch.randn((N, channels, image_size, image_size), device=self.device)
        else:
            if cond_img is not None:
                cond_img = cond_img[:N].to(self.device)
            x_start = x_start[:N].to(self.device)
            label = torch.zeros_like(label[:N]).to(self.device)
            context_mask = context_mask[:N].to(self.device)
            noise = torch.randn_like(x_start, device=self.device)
            t = torch.full((x_start.shape[0],), sample_timesteps[0], device=self.device, dtype=torch.long)
            x_t = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
            
        print("sample...", flush=True)
        with torch.no_grad():
            for t in sample_timesteps:
                x_t = self.sampe_step(x_t, cond_img, t, y=label, context_mask=context_mask, N=N, sample_type=self.guidance)
            if self.guidance == "cfg" and self.is_test:
                x_t = x_t[:N//2]

        return x_t
    
    def ddim_sample(self, N=8, x_start=None, cond_img=None, label=None, context_mask=None, sample_timesteps=None):
        image_size = self.image_size
        channels = self.channels
        if x_start is None:
            x_t = torch.randn((N, channels, image_size, image_size), device=self.device)
        else:
            # label[:N//2] = 1
            # context_mask = None
            noise = torch.randn_like(x_start, device=self.device)
            t = torch.full((x_start.shape[0],), 0, device=self.device, dtype=torch.long)
            x_t = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
            # x_t = x_start.to(self.device)
            
        if cond_img is not None:
            cond_img = cond_img[:N].to(self.device)
        if context_mask is not None:
            context_mask = context_mask[:N].to(self.device)
        label = torch.zeros_like(label[:N]).to(self.device)
        # label = torch.ones_like(label[:N]).to(self.device)
        reverse_timesteps = list(sample_timesteps)[::-1]

        print("sample...", flush=True)
        with torch.no_grad():
            for t in reverse_timesteps:
            # for i in range(0, self.infer_step):
                # t = reverse_timesteps[i]
                x_t, _ = self.sampe_step(x_t, cond_img, t, y=label, context_mask=context_mask, N=N, sample_type="inver_ddim")
                
            for t in sample_timesteps:
            # for i in range(0, self.infer_step):
                # t = sample_timesteps[i]
                x_t, _ = self.sampe_step(x_t, cond_img, t, y=label, context_mask=context_mask, N=N, sample_type=self.guidance)
            if self.guidance == "cfg":
                x_t = x_t[:N//2]
        return x_t
    
    def log_images(self, epoch, x_start, cond_img, N=2, label=None, context_mask=None):
        sample_timesteps = self.noise_scheduler.timesteps[self.num_timesteps-self.log_noise:]
        imgs = self.sample(N, x_start, cond_img, label=label,context_mask=context_mask, sample_timesteps=sample_timesteps)
        for i in range(imgs.shape[0]):
            for ch in range(imgs.shape[1]):
                img = imgs[i][ch]
                img = (img - img.min())/(img.max()-img.min())*255.0
                img = img.to(torch.uint8).cpu().numpy()
                Image.fromarray(img).convert('L').save(os.path.join(self.log_path, str(epoch)+f'_{str(ch)}_{str(i)}'+".png"))

    def metric(self, sample_imgs, x_start, seg, number, th):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        x_start = x_start.cpu().numpy()
        seg = seg.numpy()
        sample_imgs = sample_imgs.cpu().numpy()
        dice_dict = {}
        batch_dice = 0
        for i in range(sample_imgs.shape[0]):
            for ch in range(sample_imgs.shape[1]):
                # 要按每个通道操作
                # sample_imgs[i][ch] = Normalize(hist_mach(Normalize(sample_imgs[i][ch], 255), Normalize(x_start[i][ch], 255)), 1.)
                sample_imgs[i][ch] = Normalize(sample_imgs[i][ch])
            res = abs(Normalize(x_start[i]) - sample_imgs[i]).sum(axis=0)
            res_th = Normalize(res.copy())
            # res_th = cv2.erode(res_th, kernel)
            output_dir = os.path.join(self.sample_path, number[i])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # if False: 
            if seg is not None: 
                seg_i = seg[i][0]
                dice = dice_score(res_th, seg_i, th)
                batch_dice += dice
                dice_dict[number[i]] = dice
                Image.fromarray(Normalize(seg_i, 255)).convert('L').save(os.path.join(output_dir, 'seg.png'))
            
            # 保存图像
            for ch in range(sample_imgs.shape[1]):
                Image.fromarray(Normalize(x_start[i][ch], 255)).convert('L').save(os.path.join(output_dir, str(ch)+'.png'))
                Image.fromarray(Normalize(sample_imgs[i][ch], 255)).convert('L').save(os.path.join(output_dir, str(ch)+'_test.png'))
            Image.fromarray(Normalize(res, 255)).convert('L').save(os.path.join(output_dir, f'res_w={self.guide_w}.png'))
            res_color = cv2.applyColorMap(Normalize(res, 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, 'res_color.png'), res_color)
            Image.fromarray(Normalize(res_th, 255)).convert('L').save(os.path.join(output_dir, f'res_th={th}.png'))
            
        return batch_dice/sample_imgs.shape[0], dice_dict
            
    def training_step(self, batch, optim_idx, epoch, device):
        self.device = device
        self.is_test = False
        x_start, cond_img, is_fcd, seg, number, context_mask = self.get_input(batch)
        loss = self(x_start, cond_img, is_fcd, context_mask)
        return loss
    
    def testing_step(self, batch, sample_path, device, noise_level, th, eta, guide_w, infer_step):
        if self.guidance == "cg" and self.test_init==0:
            self.classifier_model = instantiate_from_config(self.classifier_config).to(device).eval()
            self.test_init = 1
            
        self.device = device
        self.is_test = True
        self.sample_path = sample_path
        self.eta = eta
        self.guide_w = guide_w
        # print("noise_level:", noise_level)
        print("infer_step:", infer_step)
        x_start, cond_img, is_fcd, seg, number, context_mask = self.get_input(batch)
        if not self.use_ddim:
            self.noise_scheduler.set_timesteps(timesteps=range(0,noise_level)[::-1], device=device)
            sample_timesteps = self.noise_scheduler.timesteps
            sample_imgs = self.sample(x_start.shape[0], x_start, cond_img, is_fcd, context_mask, sample_timesteps)
        else:
            self.infer_step = infer_step
            self.noise_scheduler.set_timesteps(self.infer_step, device=device)
            self.inverse_scheduler.set_timesteps(self.infer_step, device=device)
            sample_timesteps = self.noise_scheduler.timesteps[self.num_timesteps-noise_level:]
            # sample_timesteps = self.noise_scheduler.timesteps
            sample_imgs = self.ddim_sample(x_start.shape[0], x_start, cond_img, is_fcd, context_mask, sample_timesteps)
        
        batch_mean_dice, dice_dict = self.metric(sample_imgs, x_start, seg, number, th)
        return batch_mean_dice, dice_dict

    def end_epoch(self, epoch, batch):
        if (epoch+1) % self.log_image_epoch == 0:
            x_start, cond_img, is_fcd, seg, number, context_mask = self.get_input(batch)
            self.log_images(epoch, x_start, cond_img, label=is_fcd, context_mask=None)
    
    def get_input(self, batch):
        img, seg, is_fcd, number = batch
        if self.condition_type == "none":
            x_start = img
            cond_img = None
        elif self.condition_type == "t1":
            x = img.chunk(2, dim=1)
            x_start = x[0]
            cond_img = x[1]
        elif self.condition_type == "flair":
            x = img.chunk(2, dim=1)
            x_start = x[0]
            cond_img = x[0].clone()
            
        context_mask = None
        if self.guidance == "cfg":
            if self.is_test:
                context_mask = torch.zeros_like(is_fcd).to(self.device)
                is_fcd = is_fcd.repeat(2)
                context_mask = context_mask.repeat(2)
                context_mask[x_start.shape[0]:] = 1.
                x_start = x_start.repeat(2,1,1,1)
                if cond_img is not None:
                    cond_img = cond_img.repeat(2,1,1,1)
            else:
                context_mask = torch.bernoulli(torch.zeros_like(is_fcd)+1.0-self.drop_prob).to(self.device)
        
        x_start = x_start.to(self.device)
        if cond_img is not None:
            cond_img = cond_img.to(self.device)
        is_fcd = is_fcd.to(self.device)
        return x_start, cond_img, is_fcd, seg, number, context_mask
    
    def forward(self, x_start, cond_img, label=None, context_mask=None):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
        
        if self.use_condition:
            x_noisy = torch.cat((x_noisy, cond_img), dim=1)
        
        model_out = self.model(x_noisy, t, y=label, context_mask=context_mask)
        target = noise
        pred = model_out
        loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        loss_simple = loss.mean()
        loss = loss_simple
        return loss
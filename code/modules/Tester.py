import torch
import os
import pandas as pd
import copy
from tqdm import tqdm

class Tester:
    def __init__(self, model_path, sample_path=None, device='cuda', 
                 noise_level=250, infer_step=200, th=0.5, use_double_model=False, t1_cond_w=0.5, eta=0.5, guide_w=7.5):
        self.dataloader = None
        self.model = None
        self.device = device
        self.noise_level = noise_level
        self.eta = eta
        self.use_double_model = use_double_model
        self.t1_cond_w = t1_cond_w
        self.guide_w = guide_w
        self.infer_step = infer_step
        if use_double_model:
            self.t1_cond_checkpoint = torch.load(model_path[0])
            self.flair_cond_checkpoint = torch.load(model_path[1])
        else:
            self.checkpoint = torch.load(model_path)
        self.sample_path = f"{sample_path}_nosie={str(noise_level)}_guide_w={str(guide_w)}_eta={eta}"
        # self.sample_path = f"{sample_path}_inf_step={str(infer_step)}_guide_w={str(guide_w)}"
        self.th = th
        if not os.path.exists(self.sample_path):
            os.mkdir(self.sample_path)
        
    def setup(self, model, dataloader, classifier_model=None):
        if self.use_double_model:
            self.t1_cond_model = model
            self.flair_cond_model = copy.deepcopy(model)
            self.t1_cond_model.load_state_dict(self.t1_cond_checkpoint["state_dict"])
            self.flair_cond_model.load_state_dict(self.flair_cond_checkpoint["state_dict"])
            self.t1_cond_model.to(self.device)
            self.flair_cond_model.to(self.device)
            self.flair_cond_model.eval()
            self.t1_cond_model.eval()
        else:
            self.model = model
            self.model.load_state_dict(self.checkpoint["state_dict"])
            # self.model.load_state_dict(self.checkpoint)
            self.model.to(self.device)
            self.model.eval()
        self.dataloader = dataloader

    def double_model_testing_step(self, batch):
        self.t1_cond_model.condition_type = "t1"
        self.flair_cond_model.condition_type = "flair"
        self.t1_cond_model.device = self.device
        self.flair_cond_model.device = self.device
        self.flair_cond_model.sample_path = self.sample_path
        self.t1_cond_model.sample_path = self.sample_path
        
        x_start, t1_cond_img, is_fcd, seg, number, context_mask = self.t1_cond_model.get_input(batch.copy())
        _, flair_cond_img, is_fcd, seg, number, context_mask = self.flair_cond_model.get_input(batch)
        
        # ddim
        self.t1_cond_model.noise_scheduler.set_timesteps(self.infer_step, device=self.device)
        self.t1_cond_model.inverse_scheduler.set_timesteps(self.infer_step, device=self.device)
        sample_timesteps = self.noise_scheduler.timesteps[self.num_timesteps-self.noise_level:]
        
        noise = torch.randn_like(x_start, device=self.device)
        t = torch.full((x_start.shape[0],), 0, device=self.device, dtype=torch.long)
        x_t_t1 = self.noise_scheduler.add_noise(x_start, noise, t).to(self.device)
        x_t_flair = x_t_t1.copy()
        reverse_timesteps = list(sample_timesteps)[::-1]
        
        print("sample...", flush=True)
        with torch.no_grad():
            for t in reverse_timesteps[1:]:
            # for i in range(0, self.infer_step):
                # t = reverse_timesteps[i]
                x_t_t1, _ = self.sampe_step(x_t_t1, t1_cond_img, t, y=is_fcd, context_mask=None, N=x_start.shape[0], sample_type="inver_ddim")
                x_t_flair, _ = self.sampe_step(x_t_flair, flair_cond_img, t, y=is_fcd, context_mask=None, N=x_start.shape[0], sample_type="inver_ddim")
            
            for t in sample_timesteps:
            # for i in range(0, self.infer_step):
                # t = sample_timesteps[i]
                _, model_out_t1 = self.sampe_step(x_t_t1, t1_cond_img, t, y=is_fcd, context_mask=None, N=x_start.shape[0], sample_type="none")
                x_t_flair, model_out_flair = self.sampe_step(x_t_flair, flair_cond_img, t, y=is_fcd, context_mask=None, N=x_start.shape[0], sample_type="none")
                model_out_t1 = self.t1_cond_w * model_out_t1 + (1-self.t1_cond_w) * model_out_flair
                
                x_t_t1, model_out_t1 = self.sampe_step(x_t_t1, t1_cond_img, t, y=is_fcd, context_mask=None, N=x_start.shape[0], sample_type="cg", pre_noise=model_out_t1)
                
        batch_mean_dice, dice_dict = self.t1_cond_model.metric(x_t_t1, x_start, seg, number, self.th)
        return batch_mean_dice, dice_dict
            

    def test(self):
        if self.dataloader is None:
            print("model and dataloader must be setup")
            return
        all_step = len(self.dataloader)
        epoch_dice = 0
        all_dice_dict = {}
        for j, batch in enumerate(self.dataloader):
            with torch.no_grad():
                if self.use_double_model:
                    dice, dice_dict = self.double_model_testing_step(batch)
                else:
                    dice, dice_dict = self.model.testing_step(batch, self.sample_path, self.device, self.noise_level, self.th, self.eta, self.guide_w, self.infer_step)
                epoch_dice += dice
                all_dice_dict.update(dice_dict)
        
        df = pd.DataFrame.from_dict(all_dice_dict, orient='index',columns=['dice'])
        df = df.reset_index().rename(columns = {'index':'id'})
        df.to_csv(os.path.join(self.sample_path, "dice.csv"))
        print(f"dice: {epoch_dice/all_step}")
        
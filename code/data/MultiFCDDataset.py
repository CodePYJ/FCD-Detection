import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as Trans
import pandas as pd

class FCDDataset(torch.utils.data.Dataset):
    def __init__(self, size, path, use_cond=True, test=False, tsv_path=None):
        super().__init__()
        self.img_directory = os.path.expanduser(path)
        self.transform = Trans.Resize((size, size))
        self.test = test
        self.size = size
        self.use_cond = use_cond
        self.file_list = []
        self.lobe_dict = {}
        lobe_idx = {'FL':1, 'TL,OL':2, 'FL,PL':3, 'TL':4, 'IL':5, 'PL':6, 'TL,PL':7, 'OL':8}
        if tsv_path is not None:
            tsv = pd.read_csv(tsv_path, sep='\t')
            ids = tsv['participant_id']
            lobe = tsv['lobe']
            for i in range(len(ids)//2):
                number = ids[i].split('-')[1]
                self.lobe_dict[number] = lobe_idx[lobe[i]]
        
        for root, dirs, files in os.walk(self.img_directory):
            if not dirs:
                files.sort()
                file_dict = {}
                is_hc = False if len(files) == 3 else True
                for f in files:
                    f_split = f.split('_')
                    if 'T1w' in f_split:
                        seqtype = 't1'
                    elif 'roi' in f_split:
                        seqtype = 'seg'
                    else:
                        seqtype = 'flair'
                    file_dict[seqtype] = os.path.join(root, f)
                    if is_hc:
                        file_dict['seg'] = 0
                self.file_list.append(file_dict)

    def __getitem__(self, x):
        f_dict = self.file_list[x]
        flair_path = f_dict["flair"]
        t1_path = f_dict["t1"]

        number = flair_path.split('/')[-2]
        lobe_idx = number.split('_')[0]
        np_flair = np.array(Image.open(flair_path))
        np_flair = (np_flair - np.min(np_flair)) / (np.max(np_flair) - np.min(np_flair))
        np_flair = (np_flair - 0.5) * 2.
        
        np_t1 = np.array(Image.open(t1_path))
        np_t1 = (np_t1 - np.min(np_t1)) / (np.max(np_t1) - np.min(np_t1))
        np_t1 = (np_t1 - 0.5) * 2.
        
        if self.transform:
            flair = self.transform(torch.tensor(np_flair)[None,:])
            t1 = self.transform(torch.tensor(np_t1)[None,:])
            
        seg = torch.zeros((1,self.size,self.size))
        if self.use_cond:
            img = torch.zeros((2,self.size,self.size))
            img[1] = t1
        else:
            img = torch.zeros((1,self.size,self.size))
            
        img[0] = flair
        lobe = 0
        if f_dict['seg'] != 0:
            np_mask = np.array(Image.open(f_dict['seg']))
            seg[0] = self.transform(torch.tensor(np_mask)[None,:])
            lobe = self.lobe_dict[lobe_idx]
            
        return (img, seg, lobe, number)

    def __len__(self):
        return len(self.file_list)
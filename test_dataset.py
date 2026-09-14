import torch

from torch.utils.data import Dataset
import numpy as np



class RDataset(Dataset):
    def __init__(self, obs, acts, target, transform):
        self.obs = obs
        self.acts = acts
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        img = self.obs[idx]  # 84x84x3
        act = self.acts[idx]
        target = self.target[idx]

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        img = self.transform(img)

        act = torch.from_numpy(act).float()
        target = torch.from_numpy(target).float()
        return img, act, target



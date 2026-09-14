## Knowledge Distillation for Multimodal Visuomotor Policies

A study of the effectiveness of knowledge distillation for compressing VLA‑like manipulator control models.

--------

## Overview

The paper investigates the use of knowledge distillation to compress a multimodal visuomotor policy that takes as input a scene image and the coordinates of a target point. The teacher is a ResNet-18 with an MLP branch for the target (~11M parameters); the students are compact CNNs (~30k parameters), both multimodal and unimodal.

Experiments show that pre‑training is critically important for a multimodal learner (MAE improves by 2.5 times), while a unimodal learner can approach a multimodal one through distillation. This opens up the possibility of deploying compact policies without explicitly passing the target as input.


--------

## Quick Start

Load model (TeacherModel - for teacher, StudentModel1 - for miltimodal student, StudentModel2 - for unimodal student)

```python
import torch
from test_dataset import RDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from model import TeacherModel, StudentModel1, StudentModel2 # from model.py
import pickle
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('your_dataset.pkl', 'rb') as f: # Read file
    frames = pickle.load(f)

observ, actions, targets= [], [], []
for i in frames:
    observ.append(i['observations'])
    actions.append(i['actions'])
    targets.append(i['targets'])

observ = np.concatenate(observ, axis=0)
actions = np.concatenate(actions, axis=0)
targets = np.concatenate(targets, axis=0)

actions_mean = actions.mean(axis=0) # normalization
actions_std  = actions.std(axis=0) + 1e-8
actions_norm = (actions - actions_mean) / actions_std

targets_mean = targets.mean(axis=0)
targets_std  = targets.std(axis=0) + 1e-8
targets_norm = (targets - targets_mean) / targets_std

model = TeacherModel().to(device)
state_dict = torch.load('teacher.pth', map_location=device, weights_only=False) # Load weight
modelTeacher.load_state_dict(state_dict['state_dict'])
model.eval()

with torch.no_grad(): # Evaluation
    mae = 0
    for imgs, acts, target in val_loader:
        imgs, acts, target = imgs.to(device), acts.to(device), target.to(device)
        outputs = model_teacher(imgs, target) 
    
        y_pred = outputs.cpu().numpy()
        y_true = acts.cpu().numpy() 

        mae += mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae / 300}")
    MAE.append(mae / 300)

```
--------

## Data format

- Multimodal

```python

frames = [
    {
        'observations': np.ndarray,  # (T, 84, 84, 3), dtype=uint8, значения [0, 255]
        'actions':      np.ndarray,  # (T, 2), dtype=float32, крутящие моменты
        'targets':      np.ndarray,  # (T, 2), dtype=float32, координаты цели (x, y)
        'success':      bool,        # достиг ли эксперт цели
        'length':       int,         # T — число кадров в эпизоде
    },
    ...
]
```

- Unimodal

```python
frames = [
    {
        'observations': np.ndarray,  # (T, 84, 84, 3), dtype=uint8, значения [0, 255]
        'actions':      np.ndarray,  # (T, 2), dtype=float32, крутящие моменты
        'success':      bool,        # достиг ли эксперт цели
        'length':       int,         # T — число кадров в эпизоде
    },
    ...
]
```

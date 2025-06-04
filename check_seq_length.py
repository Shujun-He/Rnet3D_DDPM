#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
from utils import *
import os
from Diffusion import finetuned_RibonanzaNet
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
#from Network import finetuned_RibonanzaNet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="diffusion_config.yaml", help='Path to config.py')
args = parser.parse_args()


os.system('mkdir logs')
os.system('mkdir weights')

# config_name=args.config.split("/")[1]
# exit()
# print(f"Config name: {config_name}")
# exit()

#exit()



# In[2]:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# # Config

# In[3]:



config=load_config_from_yaml(args.config)

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),config,pretrained=True)#.cuda()

# state_dict=torch.load("weights/config_003.yaml_RibonanzaNet_3D.pt",map_location='cpu')
# #state_dict=torch.load("RibonanzaNet-3D-v2.pt",map_location='cpu')

# #get rid of module. from ddp state dict
# new_state_dict={}

# for key in state_dict:
#     new_state_dict[key[7:]]=state_dict[key]

# model.load_state_dict(new_state_dict)


#save to pickle
with open("train_data.pkl", "rb") as f:
    data = pickle.load(f)

# # Split train data into train/val/test¶
# We will simply do a temporal split, because that's how testing is done in structural biology in general (in actual blind tests)

# In[10]:


# Split data into train and test
all_index = np.arange(len(data['sequence']))
cutoff_date = pd.Timestamp(config.cutoff_date)
test_cutoff_date = pd.Timestamp(config.test_cutoff_date)
train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]

train_lengths = [len(data['sequence'][i]) for i in train_index]
val_lengths=[len(data['sequence'][i]) for i in test_index]

test_index= [val_lengths[i] for i in np.argsort(val_lengths)]

# In[11]:


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")
#exit()

# # Get pytorch dataset¶

# In[12]:


from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ast import literal_eval

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

from collections import defaultdict

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices=indices
        self.data=data
        #set default to 4
        self.tokens=defaultdict(lambda: 4)
        self.tokens['A']=0
        self.tokens['C']=1
        self.tokens['G']=2
        self.tokens['U']=3

        #{nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        idx=self.indices[idx]
        sequence=[self.tokens[nt] for nt in (self.data['sequence'][idx])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        #get C1' xyz
        xyz=self.data['xyz'][idx]
        xyz=torch.tensor(np.array(xyz))


        if len(sequence)>config.max_len:
            crop_start=np.random.randint(len(sequence)-config.max_len)
            crop_end=crop_start+config.max_len

            sequence=sequence[crop_start:crop_end]
            xyz=xyz[crop_start:crop_end]
        
        #center at first atom if first atom does not exit go until it does
        for i in range(len(xyz)):
            if (~torch.isnan(xyz[i])).all():
                break
        xyz=xyz-xyz[i]

        # for i in range(len(xyz)):

        #     if torch.isnan(xyz[i]).any():
        #         if i==0:
        #             xyz[i]=xyz[i+1]
        #         else:
        #             xyz[i]=xyz[i-1]

        return {'sequence':sequence,
                'xyz':xyz}


# In[13]:


train_dataset=RNA3D_Dataset(train_index,data)
val_dataset=RNA3D_Dataset(test_index,data)


# In[14]:








# In[15]:
weights = np.array(train_lengths).clip(64,768)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


train_loader=DataLoader(train_dataset,batch_size=config.batch_size,sampler=sampler, shuffle=False)
train_loader_uniform=DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False)


# # Get RibonanzaNet¶
# We will add a linear layer to predict xyz of C1' atoms

# In[ ]:





# In[16]:





#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


from tqdm import tqdm

# epochs=50
# cos_epoch=35


best_loss=np.inf
optimizer = torch.optim.Adam(model.parameters(), weight_decay=config.weight_decay, lr=config.learning_rate) #no weight decay following AF

#batch_size=1

#for cycle in range(2):

criterion=torch.nn.CrossEntropyLoss(reduction='none')

#scaler = GradScaler()

if len(args.config.split("/"))>1:
    prefix = args.config.split("/")[-1]
else:
    prefix = args.config

logger=CSVLogger(["epoch","train_loss","val_loss","val_rmsd","val_lddt"],f"logs/{prefix}_log.csv")

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision=config.mixed_precision)


schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.epochs-config.cos_epoch)*len(train_loader)//config.batch_size)

warmup_schedule=LinearWarmupScheduler(optimizer=optimizer,
                                    total_steps=config.warmup_steps*accelerator.num_processes,
                                    final_lr=config.learning_rate)



def normal_pdf(x: torch.Tensor,
               mean: float = 0.0,
               std: float  = 1.0) -> torch.Tensor:
    """
    Probability density function of a normal distribution.
    """
    var = std ** 2
    denom = (2 * 3.1415 * var)**0.5
    num   = torch.exp(- (x - mean) ** 2 / (2 * var))
    return num / denom

# def get_sample_pdf(gt_xyz):

#     centered_xyz=gt_xyz-torch.nanmean(gt_xyz,0)
#     std=centered_xyz[centered_xyz==centered_xyz].std()

#     signal_variance_schedule = (sqrt_alpha_bars.to(gt_xyz.device))**2*(std/data_std)**2
#     center = torch.abs(signal_variance_schedule-0.5).argmin().cpu()

#     x = np.linspace(0, 1, config.n_times)
#     pdf_vals = norm.pdf(x, loc=center/config.n_times, scale=0.2)
#     pdf_vals = pdf_vals / pdf_vals.max()  # Normalize the PDF 
#     pdf_vals = torch.tensor(pdf_vals).float().to(gt_xyz.device)
#     return pdf_vals

def get_sample_pdf(gt_xyz):

    centered_xyz=gt_xyz-torch.nanmean(gt_xyz,0)
    std=centered_xyz[centered_xyz==centered_xyz].std()

    signal_variance_schedule = (sqrt_alpha_bars.to(gt_xyz.device))**2*(std/data_std)**2
    center = torch.abs(signal_variance_schedule-0.5).argmin()#.cpu()

    x = torch.linspace(0, 1, config.n_times, device=gt_xyz.device)
    pdf_vals = normal_pdf(x, center/config.n_times, 0.2)
    pdf_vals = pdf_vals / pdf_vals.max()  # Normalize the PDF 
    return pdf_vals


sqrt_alpha_bars=model.sqrt_alpha_bars
data_std=model.data_std

model, optimizer, train_loader, val_loader, schedule, warmup_schedule = \
accelerator.prepare(model, optimizer, train_loader, val_loader, schedule, warmup_schedule)
#diffusion = accelerator.prepare(diffusion)
#config.warmup_steps=config.warmup_steps//(accelerator.num_processes)
best_val_loss=99999999999
# x = np.linspace(-2, 2, config.n_times)
# pdf_vals = norm.pdf(x, loc=0, scale=1)
# pdf_vals = torch.tensor(pdf_vals).float()
total_steps=0
lengths=[]

for idx, batch in enumerate(train_loader):
    #try:
    sequence=batch['sequence']#.cuda()
    gt_xyz=batch['xyz'].squeeze()
    mask=~torch.isnan(gt_xyz)

    L=sequence.shape[1]
    lengths.append(L)

lengths_uniform=[]
for idx, batch in enumerate(train_loader_uniform):
    sequence=batch['sequence']#.cuda()
    gt_xyz=batch['xyz'].squeeze()
    mask=~torch.isnan(gt_xyz)

    L=sequence.shape[1]
    lengths_uniform.append(L)

#plot and compare the two distributions in 2 subplots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(lengths, bins=50, alpha=0.5, label='Weighted Sampler', color='blue')
plt.title('Weighted Sampler Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(lengths_uniform, bins=50, alpha=0.5, label='Uniform Sampler', color='orange')
plt.title('Uniform Sampler Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{prefix}_lengths.png")

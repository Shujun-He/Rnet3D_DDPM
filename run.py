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
weights = np.array(train_lengths).clip(256,768)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


train_loader=DataLoader(train_dataset,batch_size=config.batch_size,sampler=sampler, shuffle=False)
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
for epoch in range(config.epochs):
    model.train()
    tbar=tqdm(train_loader)
    total_loss=0
    total_distogram_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        #try:
        sequence=batch['sequence']#.cuda()
        gt_xyz=batch['xyz'].squeeze()
        mask=~torch.isnan(gt_xyz)

        L=sequence.shape[1]

        pdf_vals=get_sample_pdf(gt_xyz)
        #exit()
        gt_xyz[torch.isnan(gt_xyz)]=0

        distance_matrix=calculate_distance_matrix(gt_xyz,gt_xyz)
        distogram_mask=distance_matrix==distance_matrix
        distance_matrix=distance_matrix.clip(2,39).long()

        

        gt_xyz=gt_xyz.unsqueeze(0).repeat(config.decoder_batch_size,1,1)
        #time_steps=torch.randint(0,config.n_times,size=(gt_xyz.shape[0],)).to(gt_xyz.device)
        #select time steps with torch multinomial based  on pdf_vals
        
        time_steps=torch.multinomial(pdf_vals,config.decoder_batch_size, replacement=True).to(gt_xyz.device)
        # plt.hist(time_steps.cpu().numpy(), bins=100)
        # plt.title('Histogram of Time Steps')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Frequency')
        # plt.savefig(f'sample.png')
        # exit()
        #loss_weight=(1.1-time_steps/config.n_times)
        if accelerator.distributed_type=='NO':
            noised_xyz, noise=model.make_noisy(gt_xyz, time_steps)
        else:
            noised_xyz, noise=model.module.make_noisy(gt_xyz, time_steps)

        N_cycle=np.random.randint(1,config.max_cycles+1)
        #print(N_cycle)
        if accelerator.distributed_type!='NO':
            N_cycle=accelerator.gather(torch.tensor(N_cycle).cuda())[0].item()

        #exit()
        with accelerator.autocast():
            pred_noise,distogram_pred=model(sequence,noised_xyz,time_steps,N_cycle)#.squeeze()
        #pred_xyz=aug_xyz[:,1:-1]+pred_displacements[:,1:-1]
        #exit()

        
        loss= torch.square(noise-pred_noise)#*loss_weight[:,None,None]
        loss=loss[mask.repeat(config.decoder_batch_size,1,1)].mean()
        #exit()
        
        distogram_loss=criterion(distogram_pred.squeeze()[distogram_mask],distance_matrix[distogram_mask]).mean()
        total_distogram_loss+=distogram_loss.item()


        if loss!=loss:
            stop

        
        #(loss/batch_size*len(gt_xyz)).backward()

        loss_length_scale=L**config.loss_power_scale/(100**config.loss_power_scale)
        # print(L)
        # print(len(gt_xyz))
        #print(loss_length_scale)
        accelerator.backward((loss+config.distogram_weight*distogram_loss)/config.batch_size*loss_length_scale)

        if (idx+1)%config.batch_size==0 or idx+1 == len(tbar):

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss/batch_size).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # scaler.step(optimizer)
            # scaler.update()
            if total_steps<config.warmup_steps:
                warmup_schedule.step()
            
            lr=optimizer.param_groups[0]['lr']
            #print(total_steps,lr)

            if (epoch+1)>config.cos_epoch:
                schedule.step()
            total_steps+=1
        #schedule.step()
        total_loss+=loss.item()
        
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)} Distogram Loss: {total_distogram_loss/(idx+1)}")
        #break
    # visualize_point_cloud_batch(pred_xyz)
    # visualize_point_cloud_batch(aug_xyz)

    total_loss=total_loss/len(tbar)

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    val_rmsd=0
    val_lddt=0
    #unwrapped_diffusion=accelerator.unwrap_model(diffusion)
    #unwrapped_model=accelerator.unwrap_model(model)
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()

        with torch.no_grad():
            # if accelerator.dis
            #pred_xyz=model.module.decode(sequence,torch.ones_like(sequence).long().cuda()).squeeze()
            if accelerator.distributed_type=='NO':
                pred_xyz=model.sample_euler(sequence,1,config.val_n_steps,N_cycle=config.max_cycles)[0].squeeze(0)
            else:
                pred_xyz=model.module.sample_euler(sequence,1,config.val_n_steps,N_cycle=config.max_cycles)[0].squeeze(0)
            #pred_xyz=model(sequence)[-1].squeeze()
            loss=dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)

        val_rmsd+=accelerator.gather(align_svd_rmsd(pred_xyz,gt_xyz)).mean().item()
        val_lddt+=accelerator.gather(torch.tensor(compute_lddt(pred_xyz.cpu().numpy(),gt_xyz.cpu().numpy())).to(pred_xyz.device)).mean().item()
        val_loss+=accelerator.gather(loss).mean().item()

        val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])

        #break

    val_loss=val_loss/len(tbar)
    val_rmsd=val_rmsd/len(tbar)
    val_lddt=val_lddt/len(tbar)

    print(f"val loss: {val_loss}")
    print(f"val_rmsd: {val_rmsd}")
    print(f"val_lddt: {val_lddt}")

    if accelerator.is_main_process:
        logger.log([epoch,total_loss,val_loss,val_rmsd,val_lddt])
        
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            best_preds=val_preds
            torch.save(model.state_dict(),f'weights/{prefix}_RibonanzaNet_3D.pt')
            torch.save(optimizer.state_dict(),f'weights/{prefix}_RibonanzaNet_3D_optimizer.pt')
            
    # 1.053595052265986 train loss after epoch 0
torch.save(model.state_dict(),f'weights/{prefix}_RibonanzaNet_3D_final.pt')

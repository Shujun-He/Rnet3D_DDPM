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


import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
#from Network import finetuned_RibonanzaNet

os.system('mkdir train_preds')





#exit()



# In[2]:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# # Config

# In[3]:



config=load_config_from_yaml("grid_configs/config_003.yaml")

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),config,pretrained=True)#.cuda()

state_dict=torch.load("weights/config_003.yaml_RibonanzaNet_3D_final.pt",map_location='cpu')
#state_dict=torch.load("RibonanzaNet-3D-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)


# data={
#       "sequence":train_sequences['sequence'].to_list(),
#       "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
#       "description": train_sequences['description'].to_list(),
#       "all_sequences": train_sequences['all_sequences'].to_list(),
#       "xyz": all_xyz
# }

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


# In[11]:


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")


train_lengths=[len(data['sequence'][i]) for i in train_index]

train_index= [train_index[i] for i in np.argsort(train_lengths)]



from torch.utils.data import Dataset, DataLoader
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


import plotly.graph_objects as go
import numpy as np





# In[15]:


train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=False)
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

logger=CSVLogger(["epoch","train_loss","val_loss","val_rmsd","val_lddt"],"log.csv")

schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.epochs-config.cos_epoch)*len(train_loader)//config.batch_size)


from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision='bf16')



model, optimizer, train_loader, val_loader, schedule = \
accelerator.prepare(model, optimizer, train_loader, val_loader, schedule)
#diffusion = accelerator.prepare(diffusion)

best_val_loss=99999999999

tbar=tqdm(train_loader)
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
        # if accelerator.distributed_type=='NO':
        #     pred_xyz=model.sample_euler(sequence,1,config.val_n_steps)[0].squeeze(0)
        # else:
        #     pred_xyz=model.module.sample_euler(sequence,1,config.val_n_steps)[0].squeeze(0)
        if accelerator.distributed_type=='NO':
            pred_xyz=model.sample_euler(sequence,1,config.val_n_steps)[0].squeeze(0)
        else:
            pred_xyz=model.module.sample_euler(sequence,1,config.val_n_steps)[0].squeeze(0)
        #pred_xyz=model(sequence)[-1].squeeze()
        loss=dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)

    val_rmsd+=accelerator.gather(align_svd_rmsd(pred_xyz,gt_xyz)).mean().item()
    val_lddt+=accelerator.gather(torch.tensor(compute_lddt(pred_xyz.cpu().numpy(),gt_xyz.cpu().numpy())).to(pred_xyz.device)).mean().item()
    val_loss+=accelerator.gather(loss).mean().item()

    val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])

    #save pred_xyz in gt_xyz in train_preds as a pickle file
    with open(f"train_preds/{accelerator.process_index}_{idx}.pkl", "wb") as f:
        pickle.dump([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()], f)


val_loss=val_loss/len(tbar)
val_rmsd=val_rmsd/len(tbar)
val_lddt=val_lddt/len(tbar)

if accelerator.is_main_process:


    print(f"train loss: {val_loss}")
    print(f"train rmsd: {val_rmsd}")
    print(f"train lddt: {val_lddt}")

    #write to text file as well
    with open("train_preds_metrics.txt", "w") as f:
        f.write(f"train loss: {val_loss}\n")
        f.write(f"train rmsd: {val_rmsd}\n")
        f.write(f"train lddt: {val_lddt}\n")
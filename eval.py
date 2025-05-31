import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from utils import *
import os
#from Diffusion import Diffusion
import argparse

#get target csv from argparse, config, and weight 
parser = argparse.ArgumentParser(description='Evaluate RNA structure prediction model')
parser.add_argument('--target_csv', type=str, default='../input/test_sequences.csv', help='Path to the target CSV file')
parser.add_argument('--config', type=str, default='stage2.yaml', help='Path to the configuration YAML file')
parser.add_argument('--weights', type=str, default='weights/stage2.yaml_RibonanzaNet_3D.pt', help='Path to the model weights file')

args = parser.parse_args()

test_data=pd.read_csv(args.target_csv)#.loc[2:].reset_index(drop=True)

from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}

test_dataset=RNADataset(test_data)



import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Diffusion import finetuned_RibonanzaNet





config=load_config_from_yaml(args.config)

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),config,pretrained=True).cuda()
#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


import torch
state_dict=torch.load(args.weights,map_location='cpu')
#state_dict=torch.load("RibonanzaNet-3D-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)


from tqdm import tqdm
model.eval()
preds=[]
for i in tqdm(range(len(test_dataset))):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()
    target_id=test_data.loc[i,'target_id']

    predicted_dm=[]
    #for _ in range(5):
    with torch.no_grad():
        xyz,distogram=model.sample_euler(src,5,200)

    preds.append(xyz.cpu().numpy())


ID=[]
resname=[]
resid=[]
x=[]
y=[]
z=[]

data=[]

for i in range(len(test_data)):
    #print(test_data.loc[i])

    
    for j in range(len(test_data.loc[i,'sequence'])):
        # ID.append(test_data.loc[i,'sequence_id']+f"_{j+1}")
        # resname.append(test_data.loc[i,'sequence'][j])
        # resid.append(j+1) # 1 indexed
        row=[test_data.loc[i,'target_id']+f"_{j+1}",
             test_data.loc[i,'sequence'][j],
             j+1]

        for k in range(5):
            for kk in range(3):
                row.append(preds[i][k][j][kk])
        data.append(row)

columns=['ID','resname','resid']
for i in range(1,6):
    columns+=[f"x_{i}"]
    columns+=[f"y_{i}"]
    columns+=[f"z_{i}"]


submission=pd.DataFrame(data,columns=columns)


csv_filename=args.target_csv.split('/')[-1].replace('.csv','')
submission.to_csv(f'{csv_filename}_predictions.csv',index=False)


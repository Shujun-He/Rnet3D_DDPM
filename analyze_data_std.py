
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


#save to pickle
with open("train_data.pkl", "rb") as f:
    data = pickle.load(f)

stds=[]
seq_lengths=[]
for i in tqdm(range(len(data['xyz']))):
    xyz=data['xyz'][i]

    xyz=xyz-np.nanmean(xyz,0)
    stds.append(np.std(xyz))
    seq_lengths.append(len(xyz))


plt.hist(stds, bins=50)
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.title('Histogram of Standard Deviations')
plt.savefig('std_histogram.png')
plt.clf()

plt.hist(seq_lengths, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Histogram of Sequence Lengths')
plt.savefig('seq_length_histogram.png')
plt.clf()

plt.scatter(stds, seq_lengths,alpha=0.2)
plt.xlabel('Standard Deviation')
plt.ylabel('Sequence Length')
plt.title('Scatter Plot of Standard Deviation vs Sequence Length')
plt.savefig('std_vs_seq_length.png')

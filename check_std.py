import pickle 
import numpy as np
import matplotlib.pyplot as plt

with open("train_data.pkl", "rb") as f:
    data = pickle.load(f)

seq_lengths= [len(xyz) for xyz in data['xyz']]
stds=[]
for xyz in data['xyz']:
    
    #center and take std
    #xyz[np.isnan(xyz)] = 0  # Replace NaNs with 0
    xyz = xyz - np.nanmean(xyz,axis=0)  # Center the data by subtracting the mean
    std = xyz[~np.isnan(xyz)].std()
    stds.append(std)

plt.figure(figsize=(10, 5))
plt.scatter(seq_lengths, stds, alpha=0.5)
plt.title('Standard Deviation vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Standard Deviation')
plt.grid()
plt.tight_layout()
plt.savefig("std_vs_length.png")


plt.hist(stds, bins=50, density=True)
plt.title('Histogram of Standard Deviations')
plt.xlabel('Standard Deviation')
plt.ylabel('Density')
plt.grid()
plt.tight_layout()
plt.savefig("std_histogram.png")
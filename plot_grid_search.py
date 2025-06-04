from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

start_epoch = 10
metrics = ['train_loss', 'val_loss', 'val_rmsd', 'val_lddt']

### 1. Grid Search Summary (val_lddt vs. epoch) ###
csvs = glob("logs/config*.csv")
plt.figure(figsize=(10, 6))

best_val_lddt = 0
for csv in csvs:
    df = pd.read_csv(csv).iloc[start_epoch:]
    filename = csv.split("/")[-1].split(".")[0]
    model_name = filename.split("_")[1]
    plt.plot(df["epoch"], df["val_lddt"], label=model_name)

    if df["val_lddt"].max() > best_val_lddt:
        best_val_lddt = df["val_lddt"].max()
        best_model_name = model_name

print(f"Best model: {best_model_name} with val_lddt: {best_val_lddt:.4f}")

plt.xlabel("Epoch")
plt.ylabel("Validation lDDT")
plt.title("Grid Search (val_lddt)")
plt.legend()
plt.grid(True)
plt.savefig("grid_search_val_lddt.png")
plt.close()

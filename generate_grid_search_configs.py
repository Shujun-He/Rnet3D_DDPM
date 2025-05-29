import itertools
import yaml
import copy
import os

# === Paths ===
BASE_CONFIG_PATH = "diffusion_config.yaml"
CONFIG_DIR = "grid_configs"
SLURM_DIR = "slurm_scripts"
RUN_SCRIPT = "/lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/accelerate launch run.py"

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(SLURM_DIR, exist_ok=True)


# Define search grid
beta_scales = [0.5, 1.0, 2.0]  # scale both beta_min and beta_max
distogram_weights = [0.03, 0.2]
data_stds = [17, 35]
grad_clips = [1.0, 10.0]

# === Load Base Config ===
with open(BASE_CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

# === Generate Configs and SLURM Scripts ===
combinations = list(itertools.product(beta_scales, distogram_weights, data_stds, grad_clips))

for i, (scale, dweight, std, clip) in enumerate(combinations):
    config = copy.deepcopy(base_config)
    config["beta_min"] = scale * base_config["beta_min"]
    config["beta_max"] = scale * base_config["beta_max"]
    config["distogram_weight"] = dweight
    config["data_std"] = std
    config["grad_clip"] = clip

    config_filename = f"config_{i:03d}.yaml"
    config_path = os.path.join(CONFIG_DIR, config_filename)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=3d-gridsearch-job-{i:03d}
#SBATCH --partition=defq

{RUN_SCRIPT} --config {config_path}
"""
    slurm_path = os.path.join(SLURM_DIR, f"job_{i:03d}.sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

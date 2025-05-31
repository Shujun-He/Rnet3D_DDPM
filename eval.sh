#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=eval
#SBATCH --partition defq

python eval.py --target_csv ../input/validation_sequences.csv
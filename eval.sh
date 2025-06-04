#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=eval
#SBATCH --partition defq

#python eval.py --target_csv ../input/validation_sequences.csv  --weights weights/recycle_restart.yaml_RibonanzaNet_3D.pt --config recycle.yaml
python eval.py --target_csv ../CONFIDENTIAL/rerun_DATA_REFRESH/test_sequences.csv  --weights weights/recycle.yaml_RibonanzaNet_3D.pt --config recycle.yaml
#python eval.py --target_csv ../CONFIDENTIAL/rerun_DATA_REFRESH/test_sequences.csv --weights weights/recycle.yaml_RibonanzaNet_3D.pt --config recycle.yaml
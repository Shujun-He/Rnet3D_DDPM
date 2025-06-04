#!/bin/bash

# Directory containing the SLURM scripts
SCRIPT_DIR="slurm_scripts"

# Loop over all job scripts and submit
for script in "$SCRIPT_DIR"/job_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
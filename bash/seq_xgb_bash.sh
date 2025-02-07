#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=seq_ensemble_train
#SBATCH -A clifton.prj
#SBATCH -o slurm_logs/seq_ensemble_train.out
#SBATCH -e slurm_logs/seq_ensemble_train.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate the virtual environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Load libffi module
module load libffi/3.4.4-GCCcore-12.3.0

# Run the sequential ensemble training Python script
python /well/clifton/users/ncu080/ad4mm/python/seq_xgb_train.py

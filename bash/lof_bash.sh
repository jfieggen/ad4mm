#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=lof_novelty
#SBATCH -A clifton.prj
#SBATCH --mem=32G
#SBATCH -o slurm_logs/lof_novelty.out
#SBATCH -e slurm_logs/lof_novelty.err

echo "------------------------------------------------"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Start: $(date)"
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# 1) Activate your Python environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# 2) Load modules if needed
module load libffi/3.4.4-GCCcore-12.3.0

# 3) Run the Python script
python /well/clifton/users/ncu080/ad4mm/python/lof_train.py

echo "Job finished at: $(date)"

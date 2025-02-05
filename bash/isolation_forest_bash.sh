#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=isolation_forest
#SBATCH -A clifton.prj
#SBATCH -o slurm_logs/isolation_forest.out
#SBATCH -e slurm_logs/isolation_forest.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate your Python environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Load any necessary modules
module load libffi/3.4.4-GCCcore-12.3.0

# Run your Isolation Forest Python script
python /well/clifton/users/ncu080/ad4mm/python/isolation_forest.py

echo "Job finished at: $(date)"

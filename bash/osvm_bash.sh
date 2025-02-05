#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=ocsvm
#SBATCH --mem=64G
#SBATCH -A clifton.prj
#SBATCH -o slurm_logs/ocsvm.out
#SBATCH -e slurm_logs/ocsvm.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate your Python environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Optionally load modules if needed
module load libffi/3.4.4-GCCcore-12.3.0

# Run the Python script
python /well/clifton/users/ncu080/ad4mm/python/svm_train.py

echo "Job finished at: $(date)"

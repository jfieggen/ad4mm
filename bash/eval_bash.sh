#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH -A clifton.prj
#SBATCH -o slurm_logs/eval.out
#SBATCH -e slurm_logs/eval.err

echo "------------------------------------------------"
echo "Run on host: $(hostname)"
echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started at: $(date)"
echo "------------------------------------------------"

# Activate the Python virtual environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Load necessary modules
module load libffi/3.4.4-GCCcore-12.3.0
module load bzip2/1.0.8-GCCcore-12.3.0

# Run the master evaluation script
python /well/clifton/users/ncu080/ad4mm/python/eval_script.py

echo "Job finished at: $(date)"

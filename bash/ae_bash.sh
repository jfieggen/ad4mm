#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gpus=1
#SBATCH --job-name=autoencoder_train
#SBATCH -A clifton.prj
#SBATCH -o slurm_logs/autoencoder_train.out
#SBATCH -e slurm_logs/autoencoder_train.err

echo "------------------------------------------------"
echo "Run on host: $(hostname)"
echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started at: $(date)"
echo "------------------------------------------------"

# Activate the Python virtual environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Optionally load any required modules
module load libffi/3.4.4-GCCcore-12.3.0

# Run the autoencoder training script
python /well/clifton/users/ncu080/ad4mm/python/ae_train.py

echo "Job finished at: $(date)"

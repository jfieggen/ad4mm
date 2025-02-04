#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=data_clean

# Account name and target partition
#SBATCH -A clifton.prj
#SBATCH -p short

#SBATCH --mem=16G

# Log locations which are relative to the current
# working directory of the submission
#SBATCH -o slurm_logs/data_clean.out
#SBATCH -e slurm_logs/data_clean.err

 
echo "------------------------------------------------" 
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Load libffi module
module load libffi/3.4.4-GCCcore-12.3.0

# Activate the venv environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Run your XGBoost Python script
python /well/clifton/users/ncu080/ad4mm/python/data_process.py

# End of job script
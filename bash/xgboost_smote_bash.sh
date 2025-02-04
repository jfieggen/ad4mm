#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gpus=1
#SBATCH --job-name=xgb_smote_train

#SBATCH -A clifton.prj

# Log locations which are relative to the current
# working directory of the submission
#SBATCH -o slurm_logs/xgboost_smote.out
#SBATCH -e slurm_logs/xgboost_smote.err

 
echo "------------------------------------------------" 
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate the venv environment
source /well/clifton/users/ncu080/ad4mm/ad4mm_venv/bin/activate

# Load libffi module
module load libffi/3.4.4-GCCcore-12.3.0

# Run your XGBoost Python script
python /well/clifton/users/ncu080/ad4mm/python/xgb_smote_train.py

# End of job script
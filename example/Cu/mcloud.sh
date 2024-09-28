#!/bin/sh
#SBATCH --job-name=5_off_gpu_Linear_LKF
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --partition=new3080ti,3080ti,3090

source /share/app/anaconda3/etc/profile.d/conda.sh
module load conda/3-2020.07
conda deactivate
conda activate PWMLFF
module load pwmlff/2024.5

PWMLFF train train.json

PWMLFF test test.json

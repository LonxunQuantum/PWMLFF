#!/bin/sh
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --partition=3080ti,3090

module load cuda/11.8-share
module load intel/2020

# load offline python
source /share/app/PWMLFF/offline_pwmlff/PWMLFF-2024.5/pwmlff-2024.5/bin/activate

# load offline pwmlff
export PYTHONPATH=/share/app/PWMLFF/offline_pwmlff/PWMLFF-2024.5/PWMLFF/src:$PYTHONPATH

export PATH=/share/app/PWMLFF/offline_pwmlff/PWMLFF-2024.5/PWMLFF/src/bin:$PATH

# for mcloud offline lammps app
export PATH=/share/app/PWMLFF/offline_pwmlff/PWMLFF-2024.5/lammps-2024.5/src:$PATH

export LD_LIBRARY_PATH=/share/app/PWMLFF/offline_pwmlff/PWMLFF-2024.5/lammps-2024.5/src:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import torch; print(torch.__path__[0])')/lib:$(dirname $(dirname $(which python3)))/lib:$(dirname $(dirname $(which PWMLFF)))/op/build/lib

#run 
PWMLFF train train.json

PWMLFF test test.json

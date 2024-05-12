#!/bin/bash

#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --constraint=a100  # if you want a particular type of GPU

module load python cuda
source /mnt/home/yxu10/env/bin/activate
cd $SLURM_SUBMIT_DIR

python GWTC3_PV2.py 
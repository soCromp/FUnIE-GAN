#!/bin/bash -l

#SBATCH -p research
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

conda activate funie
python train_funiegan.py

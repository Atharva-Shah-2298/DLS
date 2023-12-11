#!/bin/bash

#SBATCH -J YOLO
#SBATCH -p gpu
#SBATCH -A r00160
#SBATCH --mail-type=ALL
#SBATCH --mail-user=athshah@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00

# Load the Python module
module load python/gpu/3.10.10

# Run your program
srun ./N/u/athshah/BigRed200/Breast_model.py


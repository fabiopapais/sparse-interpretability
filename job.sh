#!/bin/bash
#SBATCH --output=logs/output_file_%j.log  # Standard output file (%j is replaced with the job ID)
#SBATCH --error=logs/error_file_%j.log   # Standard error file
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Request 8 CPUs per task
#SBATCH --gpus-per-task=1        # Request 1 GPU
module load Python3.10 freeglut
source .sparse_ae/bin/activate

date
echo "Starting execution..."
time (python script.py) # generate chunks
# cd sparse-interpretability/ && time (python ./edited_sparse_coding_files/basic_l1_sweep.py) # train autoencoder
echo "Completed execution!"
date

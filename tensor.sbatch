#!/bin/bash
#SBATCH --job-name [nall-lab]tensor-flow
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus tesla:1

date
singularity exec --nv output/tensorflow-gpu.sif python3 main.py
date


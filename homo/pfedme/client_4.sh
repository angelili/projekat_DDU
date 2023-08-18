#!/bin/bash
#SBATCH --job-name=client_4
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --time=24:00:00 #the maximum walltime in format h:m:s
#SBATCH --nodelist=n01
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

srun python3 client.py

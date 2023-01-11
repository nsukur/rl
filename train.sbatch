#!/bin/bash
#SBATCH --job-name=rl-ferm
#SBATCH --partition=cuda
#SBATCH --nodes=10
#SBATCH --exclude=n17,n18
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n16
#SBATCH --time=72:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL
#SBATCH --exclusive

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
srun python rl.py

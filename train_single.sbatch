#!/bin/bash
#SBATCH --job-name=rl-ferm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --nodelist=n10
#SBATCH --exclude=n17,n18
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=240:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL
#SBATCH --exclusive

export PYTHONUNBUFFERED=1
srun python rl.py --single

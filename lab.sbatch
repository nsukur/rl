#!/bin/bash
#SBATCH --job-name=fermatlab
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --nodelist=n16
###SBATCH --exclude=n01
###SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

/usr/bin/ssh -N -f -R 28003:localhost:28003 milosevicn@axiom.pmf.uns.ac.rs
/home/milosevicn/miniconda3/envs/rl/bin/jupyter lab --port=28003

#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 1
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

module load uv
cd /mnt/home/chainje/spxperiments
srun uv run scripts/compact.py -i $SLURM_PROCID -N $SLURM_NTASKS

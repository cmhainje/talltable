#!/bin/bash
#SBATCH -J build_wcs
#SBATCH -C icelake
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 48
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

set -e

module load uv
cd /mnt/home/chainje/spxperiments/scripts
uv run python build_wcs.py -n ${SLURM_CPUS_PER_TASK}


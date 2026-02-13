#!/bin/bash
#SBATCH -J rosette
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 48
#SBATCH --cpus-per-task 1
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

module load uv
cd /mnt/home/chainje/spxperiments/scripts
# uv run find_images.py "06h33m45s" "+04d59m54s" 5.0 rosette-files.txt
srun uv run ingest.py rosette-files.txt
srun uv run compact.py

#!/bin/bash
#SBATCH -J w44+45+46
#SBATCH -C icelake
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 48
#SBATCH --cpus-per-task 1
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

set -e
WEEKS=(44 45 46)

module load uv
cd /mnt/home/chainje/spxperiments/scripts

if [ ! -f "/mnt/sdceph/users/spherex/talltable/waves.parquet" ]; then
    uv run python build_waves.py
fi

for WEEK in "${WEEKS[@]}"; do
    FILE=/mnt/home/chainje/spxperiments/slurm/files/w${WEEK}-files.txt
    if [ ! -f "$FILE" ]; then
        echo "finding all files from week ${WEEK}. this could take a while..."
        find /mnt/sdceph/users/spherex/spherex_data_qr2/level2/ -maxdepth 4 -mindepth 4 -name "*2025W${WEEK}*.fits" > $FILE
    else
        echo "loading week ${WEEK} file list..."
    fi
    echo "found $(cat $FILE | wc -l) files"
    
    srun --kill-on-bad-exit=1 uv run python ingest.py $FILE
    srun --kill-on-bad-exit=1 uv run python compact.py
    uv run python post_compact.py
done

#!/bin/bash
#SBATCH -J week21
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 48
#SBATCH --cpus-per-task 1
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

WEEK=21
FILE=w${WEEK}-files.txt

module load uv
cd /mnt/home/chainje/spxperiments/scripts

if [ ! -f "/mnt/sdceph/users/chainje/spxdb-test/waves.parquet" ]; then
    uv run python build_wave.py
fi

if [ ! -f "$FILE" ]; then
    echo "finding all files from week ${WEEK}. this could take a while..."
    find /mnt/sdceph/users/spherex/spherex_data_qr2/level2/ -maxdepth 4 -mindepth 4 -name "*2025W${WEEK}*" > $FILE
else
    echo "loading week ${WEEK} file list..."
fi
echo "found $(cat $FILE | wc -l) files"

srun uv run python ingest.py $FILE
srun uv run python compact.py

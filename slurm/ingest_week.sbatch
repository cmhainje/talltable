#!/bin/bash
#SBATCH -C icelake
# 96 tasks over 4 nodes rather than 2: each node's I/O goes through its own ceph-fuse daemon, which
# saturates at ~24 tasks (profiled 2026-10-16, scripts/profile_compact.py). Keep 96 tasks: ingest
# writes a chunk file per task per 24 images, and more tasks mean more, smaller slices to read back.
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 24
#SBATCH --cpus-per-task 1
#SBATCH -t 1-12:00:00
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

set -e
: "${YEAR:?}" "${WEEK:?}"

if [[ ! "$WEEK" =~ ^[0-9]{2}$ ]]; then
    echo "WEEK must be zero-padded to 2 digits (e.g. 05, not 5); got '$WEEK'" >&2
    exit 1
fi

module load uv
cd /mnt/home/chainje/spxperiments/scripts

if [ ! -f "/mnt/sdceph/users/spherex/talltable/waves.parquet" ]; then
    uv run python build_waves.py
fi

# the data directory of the release being ingested: TALLTABLE_DATA_DIR from the environment (auto_ingest.py
# sets it per release), otherwise from .env (the QR2 directory, for manual runs)
DATA_DIR=$(uv run python -c "from talltable.paths import require_env; print(require_env('TALLTABLE_DATA_DIR'))")
echo "data directory: ${DATA_DIR}"

FILE=/mnt/home/chainje/spxperiments/slurm/files/${YEAR}w${WEEK}-files.txt
if [ ! -f "$FILE" ]; then
    echo "finding all files from ${YEAR} week ${WEEK}. this could take a while..."
    find "${DATA_DIR}/level2/" -maxdepth 4 -mindepth 4 -name "*${YEAR}W${WEEK}_*.fits" > $FILE
else
    echo "loading week ${WEEK} file list..."
fi
echo "found $(cat $FILE | wc -l) files"

srun --kill-on-bad-exit=1 uv run python ingest.py $FILE
srun --kill-on-bad-exit=1 uv run python compact.py
uv run python post_compact.py

# keep the WCS table current (it adds a row for every image in the images table that lacks one). It is a side
# product: the database is already consistent at this point, so a failure here is reported as a warning
# (which auto_ingest.py records) rather than failing the job, which would halt automatic ingestion
uv run python build_wcs.py -n "${SLURM_CPUS_ON_NODE:-16}" || echo "WARNING: build_wcs.py failed (exit status $?); wcs.parquet was not updated"


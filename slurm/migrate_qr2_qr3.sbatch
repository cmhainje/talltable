#!/bin/bash
#SBATCH -J migrate-qr2-qr3
#SBATCH -C icelake
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 48
#SBATCH --cpus-per-task 1
#SBATCH -t 36:00:00
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

# Migrates the pixel database from the QR2 to the QR3 (R7) calibration. See
# scripts/migrate_QR2_QR3.py. Before submitting: commit the script and `git pull` on popeye
# (the script refuses to run otherwise), and pause the server.
#
#   1. `prepare` runs once, on the batch node: checks the calibration products and writes
#      the upgrade table. It never touches the partitions.
#   2. `migrate` runs in every task: task i takes partitions i, i+N, i+2N, ... (N =
#      SLURM_NTASKS). Restartable: partitions that are already migrated are skipped.
#
# Extra flags for either step go in PREPARE_ARGS / MIGRATE_ARGS, e.g. to try two partitions:
#   sbatch --export=ALL,MIGRATE_ARGS="--parts 4350605 69279763" slurm/migrate_qr2_qr3.sbatch
# (--dry-run is accepted by both steps.)

set -e

module load uv
cd /mnt/home/chainje/spxperiments

uv run python scripts/migrate_QR2_QR3.py prepare ${PREPARE_ARGS:-}
srun --kill-on-bad-exit=1 uv run python scripts/migrate_QR2_QR3.py migrate ${MIGRATE_ARGS:-}

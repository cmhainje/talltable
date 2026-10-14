#!/bin/bash
#SBATCH -J migrate-qr2-qr3-pilot
#SBATCH -C icelake
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH -t 02:00:00
#SBATCH -o /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH -e /mnt/home/chainje/spxperiments/slurm/slurm-%x-%j.out
#SBATCH --mail-type ALL
#SBATCH --mail-user ch4407@nyu.edu

# Pilot for the QR2 -> QR3 migration (scripts/migrate_QR2_QR3.py): the real thing, on a few real partitions,
# with backups, so it can be checked against the originals before the full run (slurm/migrate_qr2_qr3.sbatch).
#
#   1. picks N_PARTS partitions (default: one per task; or set PARTS="1048576 69279763 ..."), which must not
#      have been migrated yet
#   2. backs them up to $BACKUP and checks each backup byte-for-byte
#   3. `prepare` (writes the real upgrade table; harmless to repeat) and `migrate` on just those partitions
#   4. scripts/verify_migration.py compares each migrated partition with its backup, row group by row group
#
# This DOES modify the real database: it writes the upgrade table and rewrites the chosen partitions in place.
# If anything looks wrong, the printed commands at the end restore them. Commit the script and `git pull` on popeye first.
#
# Env vars: BACKUP (default /mnt/sdceph/users/spherex/talltable_pilot_backup), N_PARTS, PARTS, PILOT_TASKS.

set -euo pipefail

module load uv
cd /mnt/home/chainje/spxperiments

BACKUP=${BACKUP:-/mnt/sdceph/users/spherex/talltable_pilot_backup}
NTASKS=${PILOT_TASKS:-${SLURM_NTASKS:-8}}
N_PARTS=${N_PARTS:-$NTASKS}
DB=$(uv run python -c "from talltable.paths import DB_DIR; print(DB_DIR)")
echo "database: $DB"
echo "backup:   $BACKUP"

if [ -e "$BACKUP" ]; then
    echo "$BACKUP already exists. Refusing to overwrite a backup: remove it by hand if you are sure." >&2
    exit 1
fi

if [ -z "${PARTS:-}" ]; then
    PARTS=$(uv run python - "$DB/parts.txt" "$N_PARTS" <<'PY'
import random, sys
parts = open(sys.argv[1]).read().split()
print(" ".join(random.Random(0).sample(parts, int(sys.argv[2]))))
PY
)
fi
echo "partitions: $PARTS"

echo; echo "### the chosen partitions must not be migrated already (a backup of a migrated partition is no use)"
DRY=$(uv run python scripts/migrate_QR2_QR3.py migrate --dry-run --parts $PARTS)
echo "$DRY" | tail -n 1
if echo "$DRY" | grep -q "already migrated" || ! echo "$DRY" | grep -q "would be migrated"; then
    echo "some of these partitions are already migrated (or none would be); choose others with PARTS=..." >&2
    exit 1
fi

echo; echo "### backing up"
for p in $PARTS; do mkdir -p "$BACKUP/pixels/part=$p"; done
printf '%s\n' $PARTS | xargs -P "$NTASKS" -I{} cp "$DB/pixels/part={}/compacted.parquet" "$BACKUP/pixels/part={}/"
printf '%s\n' $PARTS | xargs -P "$NTASKS" -I{} sh -c 'cmp -s "$1/pixels/part=$3/compacted.parquet" "$2/pixels/part=$3/compacted.parquet" || { echo "BACKUP MISMATCH for part=$3" >&2; exit 255; }' _ "$DB" "$BACKUP" {}
echo "backed up and byte-compared $(echo $PARTS | wc -w) partitions"

echo; echo "### prepare"
uv run python scripts/migrate_QR2_QR3.py prepare

echo; echo "### migrate (${NTASKS} tasks)"
srun -n "$NTASKS" --kill-on-bad-exit=1 uv run python scripts/migrate_QR2_QR3.py migrate --parts $PARTS

echo; echo "### verify against the backups"
uv run python scripts/verify_migration.py --backup "$BACKUP" --row-groups 8 $PARTS

echo; echo "PILOT PASSED. Backups are in $BACKUP (delete them when you are done)."
echo "To undo the pilot instead, for each partition P:  cp $BACKUP/pixels/part=P/compacted.parquet $DB/pixels/part=P/compacted.parquet"
echo "and delete $DB/qr2_to_qr3.parquet if you are abandoning the migration."

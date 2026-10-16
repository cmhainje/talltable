"""Checks migrated partitions against backups of their originals (see slurm/migrate_qr2_qr3_pilot.sbatch).

usage: python scripts/verify_migration.py --backup <dir with pixels/part=N/compacted.parquet> [--row-groups 8] <part> [<part> ...]

The migrated partitions are read from the database in TALLTABLE_DB_DIR. For each partition, `n` row groups spread
across the file are compared with the backed-up original. The reference is deliberately naive and independent of
the migration's code path: the waveid is decoded by hand; c, delta, and zodi_scale come from the upgrade table; the
nonfunc flag comes from the calibration product. Exits nonzero if anything differs.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from talltable.paths import PIXEL_DB_PATH

spec = importlib.util.spec_from_file_location("migrate", Path(__file__).resolve().parent / "migrate_QR2_QR3.py")
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)


def mismatches(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    return int((~((got == want) | (np.isnan(got) & np.isnan(want)))).sum())


def verify_partition(part, backup, n_groups, c, delta, zodi_scale, nonfunc):
    """returns a list of problems (empty if the partition checks out)"""
    old_path = backup / "pixels" / f"part={part}" / "compacted.parquet"
    new_path = PIXEL_DB_PATH / f"part={part}" / "compacted.parquet"
    old, new = pq.ParquetFile(old_path), pq.ParquetFile(new_path)
    problems = []

    if old.metadata.num_rows != new.metadata.num_rows:
        problems.append("row count differs")
    if not old.schema_arrow.equals(new.schema_arrow, check_metadata=False):
        problems.append("schema differs")
    if migrate.has_been_migrated(old_path) or not migrate.has_been_migrated(new_path):
        problems.append("migration marker wrong")
    if sorted(p.name for p in new_path.parent.iterdir()) != ["compacted.parquet"]:
        problems.append("stray files in the partition directory")
    sizes = [new.metadata.row_group(i).num_rows for i in range(new.metadata.num_row_groups)]
    if max(sizes) > migrate.ROW_GROUP_SIZE:
        problems.append("row group larger than ROW_GROUP_SIZE")

    per = old.metadata.row_group(0).num_rows // migrate.ROW_GROUP_SIZE  # new row groups per old one
    bad = dict.fromkeys(("hphigh", "imageid", "waveid", "flux", "variance", "zodi", "flags"), 0)
    for k in np.unique(np.linspace(0, old.metadata.num_row_groups - 1, n_groups).astype(int)):
        before = old.read_row_group(int(k))
        after = pa.concat_tables(
            [new.read_row_group(j) for j in range(int(k) * per, min(int(k) * per + per, len(sizes)))]
        )
        if before.num_rows != after.num_rows:
            problems.append(f"row group {k}: {before.num_rows} rows became {after.num_rows}")
            continue
        column = lambda table, name: table[name].to_numpy()
        waveid = column(before, "waveid").astype(np.int64)
        idx = (((waveid >> 24) & 0xF) - 1) * 2040 * 2040 + ((waveid >> 12) & 0xFFF) * 2040 + (waveid & 0xFFF)
        cc, dd, ss = (a[idx].astype(np.float64) for a in (c, delta, zodi_scale))
        expected = {
            "flux": column(before, "flux").astype(np.float64) * cc + dd,
            "variance": column(before, "variance").astype(np.float64) * cc * cc,
            "zodi": column(before, "zodi").astype(np.float64) * ss,
            "flags": column(before, "flags") | (nonfunc[idx].astype(np.int32) << 6),
        }
        for name in bad:
            want = column(before, name)
            if name in expected:
                want = expected[name].astype(want.dtype)
            bad[name] += mismatches(column(after, name), want)
    if any(bad.values()):
        problems.append(f"mismatching values {bad}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--row-groups", type=int, default=8, help="row groups to compare per partition")
    ap.add_argument("parts", type=int, nargs="+")
    args = ap.parse_args()

    migrate.check_upgrade_table()
    c, delta, zodi_scale = migrate.read_upgrade_table()
    paths = {"qr2": migrate.QR2_DIR_POPEYE, "qr3": migrate.QR3_DIR_POPEYE}
    nonfunc = migrate.read_all_nonfunc(paths)

    failed = []
    for part in args.parts:
        problems = verify_partition(part, args.backup, args.row_groups, c, delta, zodi_scale, nonfunc)
        print(f"part={part}: {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}", flush=True)
        if problems:
            failed.append(part)
    print("ALL PASS" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

import argparse
import functools
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from talltable import rowcoldet_to_waveid, waveid_to_rowcoldet
from talltable.constants import ALL_COL, ALL_ROW
from talltable.paths import DB_DIR, PART_DB_PATH, PIXEL_DB_PATH

QR2_DIR_POPEYE = "/mnt/sdceph/users/spherex/spherex_data_qr2"
QR3_DIR_POPEYE = "/mnt/sdceph/users/spherex/spherex_data_qr3"
QR2_DIR_LOCAL = "/Users/connor/research/talltable/data/qr2"
QR3_DIR_LOCAL = "/Users/connor/research/talltable/data/qr3"

S3_BUCKET = "nasa-irsa-spherex"
S3_REGION = "us-west-2"

FLUX_CORR = "{qr3}/l3_flux_corrections/cal-flxc-v1-2026-191/"
GAIN_MATRIX = "{qr3}/abs_gain_matrix/cal-agm-v8-2026-191/"
DARK_R6 = "{qr2}/dark/cal-drk-v4-2025-161/"
DARK_R7 = "{qr3}/dark/cal-drk-v6-2026-191/"
NONFUNC_MAP = "{qr3}/nonfunc_v2/base-2026-191/"
ZODI_SCALING = "{qr3}/zodi_scaling_v2/cal-zodi-v2-2026-254/"
SPECTRAL_WCS = "{qr3}/spectral_wcs/cal-swcs-v5-2026-191/"

UPGRADE_TABLE_PATH = DB_DIR / "qr2_to_qr3.parquet"

N_PIX = 2040  # each detector is N_PIX x N_PIX
DETECTORS = (1, 2, 3, 4, 5, 6)
NONFUNC_BORDER = 4  # the nonfunc maps are 2048x2048, including the reference-pixel border
FLIP_X_DETECTORS = (4, 5, 6)  # nonfunc maps are in detector coordinates; these are mirrored in X

NONFUNC_FLAG = 1 << 6  # FLAGS bit 6
ROW_GROUP_SIZE = 256 * 1024  # rows per row group in migrated partitions (they were ~1M)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument(
        "--download",
        action="store_true",
        help="download missing calibration products from S3. (always on with --local; "
        "otherwise a missing product is an error)",
    )
    ap.add_argument(
        "--parts",
        type=int,
        nargs="+",
        help="migrate only these partitions (default: every partition in parts.txt)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="read-only: download and check the calibration data, report which "
        "partitions would be migrated, but write nothing",
    )
    return ap.parse_args()


def download_if_missing(template, paths, download=True):
    """checks if the data product is present. if not, downloads from S3.
    (if `download` is False, raises FileNotFoundError instead of downloading.)

    `template` is one of the product constants above (e.g. FLUX_CORR); `paths` maps
    "qr2"/"qr3" to the local base dirs. the local layout mirrors the bucket, so the
    S3 key is the template with "qr2"/"qr3" filled in as themselves.

    "present" means every file under the S3 prefix exists locally with the same size
    (so an interrupted download is detected and resumed, not mistaken for complete).
    """
    path = Path(template.format(**paths))
    prefix = template.format(qr2="qr2", qr3="qr3")
    s3 = boto3.client(
        "s3", config=Config(signature_version=UNSIGNED), region_name=S3_REGION
    )

    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix)
    objects = [o for page in pages for o in page.get("Contents", []) if o["Size"] > 0]

    # an empty (or mistyped) prefix would make the check below pass vacuously
    if not objects:
        raise FileNotFoundError(f"nothing found at s3://{S3_BUCKET}/{prefix}")

    todo = []
    for obj in objects:
        local = path / obj["Key"].removeprefix(prefix)
        if not local.is_file() or local.stat().st_size != obj["Size"]:
            todo.append((obj["Key"], local))
    if not todo:
        return

    if not download:
        raise FileNotFoundError(
            f"calibration product {path} is incomplete: {len(todo)} of {len(objects)} "
            f"file(s) are missing or have the wrong size (e.g. {todo[0][1]}). "
            "pass --download to fetch them from S3."
        )

    print(f"downloading {len(todo)} file(s): s3://{S3_BUCKET}/{prefix} -> {path}")
    for key, local in todo:
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local))  # writes a temp file, then renames


def product_file(template, paths, det):
    """path of the (single) file for detector `det` in a product directory"""
    files = sorted((Path(template.format(**paths)) / str(det)).iterdir())
    if len(files) != 1:
        raise RuntimeError(f"expected 1 file for D{det} in {template}, found {files}")
    return files[0]


def read_image(template, paths, det, ext):
    """reads one extension of a product's detector file as a native float64 array"""
    data = fits.getdata(product_file(template, paths, det), ext)
    return data.astype(np.float64)


def read_nonfunc(paths, det):
    """reads the nonfunc map for `det`, cropped and flipped to match the images.

    returns a (N_PIX, N_PIX) boolean array, True where the pixel is nonfunctional.
    """
    data = fits.getdata(product_file(NONFUNC_MAP, paths, det), "COMBINED")
    b = NONFUNC_BORDER
    nonfunc = data[b:-b, b:-b] != 0
    if det in FLIP_X_DETECTORS:
        nonfunc = nonfunc[:, ::-1]
    return nonfunc


def waveid_to_index(waveid):
    """index of `waveid` in the flat arrays returned by massage_data.

    those arrays are ordered by detector, then row, then column (the same order as
    waves.parquet), so this is just (det - 1) * N_PIX**2 + row * N_PIX + col.
    """
    row, col, det = waveid_to_rowcoldet(np.asarray(waveid))
    # the waveid has 12 bits each for row and col, so a bad waveid could otherwise
    # silently index into a different pixel
    if (row >= N_PIX).any() or (col >= N_PIX).any() or (det < 1).any() or (det > len(DETECTORS)).any():
        raise ValueError("waveid out of range (row/col >= N_PIX or detector not in 1-6)")
    return (det.astype(np.int64) - 1) * N_PIX**2 + row * N_PIX + col


def massage_data(paths):
    """read the data products and produce arrays that are easier to use

    in particular, given the data products, compute per distinct waveid
    - c:     FLUX_CORR
    - delta: AGM_v8 * (DARK_v4 - DARK_v6)
    - zodi_scale: zodi_scaling_v2, interpolated/extrapolated
        interpolation should be linear in wavelength space per-detector
        extrapolation should clamp to the table (use nearest value)
        see notebooks/popeye/zodi_scaling_v2_validation.ipynb for evidence.
        wavelengths are the pixel CWAVE from SPECTRAL_WCS (v5).
    - nonfunc: nonfunctional pixel map (boolean array)

    these are flat arrays (N_DET * N_PIX**2), ordered by detector, row, column.
    index them with waveid_to_index(waveid).

    c, delta, and zodi_scale are float32: these are the values that get written to
    the upgrade table, and the migration must apply exactly the stored values so that
    the downgrade, (flux - delta) / c, inverts it.

    note that nonfunc calibration product is in detector coordinates,
    so D4-6 are flipped along X relative to the sky.

    return c, delta, zodi_scale, nonfunc
    """
    shape = (len(DETECTORS), N_PIX, N_PIX)
    c = np.empty(shape, dtype=np.float32)
    delta = np.empty(shape, dtype=np.float32)
    zodi_scale = np.empty(shape, dtype=np.float32)
    nonfunc = np.empty(shape, dtype=bool)

    for i, det in enumerate(DETECTORS):
        c_det = read_image(FLUX_CORR, paths, det, "IMAGE")
        if not (np.isfinite(c_det) & (c_det > 0)).all():
            raise ValueError(f"D{det}: FLUX_CORR has non-finite or non-positive values")
        c[i] = c_det

        gain = read_image(GAIN_MATRIX, paths, det, "IMAGE")
        dark_r6 = read_image(DARK_R6, paths, det, "IMAGE")
        dark_r7 = read_image(DARK_R7, paths, det, "IMAGE")
        delta[i] = gain * (dark_r6 - dark_r7)

        wave = read_image(SPECTRAL_WCS, paths, det, "CWAVE")
        table = Table.read(product_file(ZODI_SCALING, paths, det), format="ascii.ecsv")
        lam, scale = np.asarray(table["wavelength_um"]), np.asarray(table["ZodiScale"])
        if np.any(np.diff(lam) <= 0):
            raise ValueError(f"D{det}: zodi scaling wavelengths are not increasing")
        zodi_scale[i] = np.interp(wave, lam, scale)  # np.interp clamps outside the table

        nonfunc[i] = read_nonfunc(paths, det)

    for name, array in (("delta", delta), ("zodi_scale", zodi_scale)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} has non-finite values")

    return c.ravel(), delta.ravel(), zodi_scale.ravel(), nonfunc.ravel()


@functools.cache
def provenance():
    """metadata describing this migration, for the footer of everything it writes.

    cached, so every file written by one run carries the same timestamp.
    """
    here = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    products = {
        name: Path(template).name
        for name, template in (
            ("flux_corr", FLUX_CORR),
            ("gain_matrix", GAIN_MATRIX),
            ("dark_qr2", DARK_R6),
            ("dark_qr3", DARK_R7),
            ("nonfunc_map", NONFUNC_MAP),
            ("zodi_scaling", ZODI_SCALING),
            ("spectral_wcs", SPECTRAL_WCS),
        )
    }
    return {
        "migration": "qr2_to_qr3",
        "release": "qr3",
        "migrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "talltable_git": commit,
        "calibration": json.dumps(products, sort_keys=True),
    }


def check_script_committed():
    """fails unless this script is tracked by git with no uncommitted changes.

    the commit hash in provenance() is only meaningful if the code that ran is the code
    at that commit. `git status --porcelain` prints a line for an untracked (??) or
    modified file, and nothing for a clean tracked one.
    """
    script = Path(__file__).resolve()
    status = subprocess.run(
        ["git", "-C", str(script.parent), "status", "--porcelain", "--", str(script)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"{script.name} is untracked or has uncommitted changes ({status!r}).\n"
            "commit it (and `git pull` on the machine you run this on) so that the "
            "recorded commit hash identifies the code that ran."
        )


def write_upgrade_table(c, delta, zodi_scale):
    """makes a new table in DB_DIR called qr2_to_qr3.parquet
    schema should be:
        waveid
        c: FLUX_CORR
        delta: AGM_v8 * (DARK_v4 - DARK_v6)
        zodi_scale: zodi_scaling_v2_D{n} inter-/extrapolated onto this waveid

    upgrade: flux_new = c * flux_old + delta;  variance_new = c**2 * variance_old;
             zodi_new = zodi_scale * zodi_old
    downgrade: flux_old = (flux_new - delta) / c, and so on.

    rows are sorted by waveid, in the same order as the arrays from massage_data.
    written to a temporary file and renamed, so the table is never half-written.
    """
    waveid = np.concatenate(
        [rowcoldet_to_waveid(ALL_ROW, ALL_COL, det) for det in DETECTORS]
    )
    if not np.array_equal(waveid_to_index(waveid), np.arange(len(waveid))):
        raise RuntimeError("waveid order does not match the order of the data arrays")

    table = pa.table({"waveid": waveid, "c": c, "delta": delta, "zodi_scale": zodi_scale})
    table = table.replace_schema_metadata(provenance())
    sorting = pq.SortingColumn.from_ordering(table.schema, [("waveid", "ascending")])

    tmp_path = UPGRADE_TABLE_PATH.with_name(UPGRADE_TABLE_PATH.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pq.write_table(
            table,
            f,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            sorting_columns=sorting,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, UPGRADE_TABLE_PATH)


def has_been_migrated(path) -> bool:
    """checks the footer of the parquet file at `path` for the migration marker.

    the marker is the provenance() metadata, which migrate_partition writes into the
    new file *before* renaming it into place, so the data and its marker commit
    together (there is no window in which a partition is migrated but unmarked).
    """
    footer = pq.read_metadata(path).metadata or {}
    return footer.get(b"migration") == b"qr2_to_qr3"


def migrate_batch(batch, c, delta, zodi_scale, nonfunc):
    """applies the upgrade to one record batch of a partition and returns the new batch.

    arithmetic is done in float64 and cast back to the column's float32.
    """
    for name in ("waveid", "flux", "variance", "flags", "zodi"):
        if batch[name].null_count:
            raise ValueError(f"column {name!r} has nulls")
    column = lambda name: batch[name].to_numpy(zero_copy_only=False)

    idx = waveid_to_index(column("waveid"))
    c_ = c[idx].astype(np.float64)
    new = {
        "flux": column("flux").astype(np.float64) * c_ + delta[idx],
        "variance": column("variance").astype(np.float64) * c_ * c_,
        "flags": column("flags") | (nonfunc[idx].astype(np.int32) * NONFUNC_FLAG),
        "zodi": column("zodi").astype(np.float64) * zodi_scale[idx],
    }
    arrays = [
        pa.array(new[f.name], type=f.type) if f.name in new else batch[f.name]
        for f in batch.schema
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=batch.schema)


def migrate_partition(part_number, c, delta, zodi_scale, nonfunc, dry_run=False):
    """migrates partition `part_number`.

    returns "migrated" (or, with dry_run, "would be migrated"), "already migrated" if
    the partition's footer says so, or "missing" if it has no compacted.parquet.

    the new file is streamed to `migrating.parquet` (same schema and writer options as
    compact.py, row groups of ROW_GROUP_SIZE, same row order so it stays sorted by
    hphigh), with provenance() in the footer. it is verified and fsynced, then renamed
    over compacted.parquet. if verification fails, migrating.parquet is left in place
    and nothing else is touched.
    """
    path = PIXEL_DB_PATH / f"part={part_number}" / "compacted.parquet"
    if not path.exists():
        tqdm.write(f"warning: part={part_number} has no compacted.parquet; skipping")
        return "missing"
    if has_been_migrated(path):
        return "already migrated"
    if dry_run:
        return "would be migrated"

    tmp_path = path.with_name("migrating.parquet")
    src = pq.ParquetFile(path)
    schema = src.schema_arrow

    # footer: keep whatever is there (apart from the schema pyarrow adds itself), add provenance
    footer = {
        k.decode(): v.decode()
        for k, v in (schema.metadata or {}).items()
        if k != b"ARROW:schema"
    }
    footer.update(provenance())
    sorting = pq.SortingColumn.from_ordering(schema, [("hphigh", "ascending")])

    n_rows, last_hphigh = 0, None
    with open(tmp_path, "wb") as f:
        with pq.ParquetWriter(
            f,
            schema.with_metadata(footer),
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            sorting_columns=sorting,
        ) as writer:
            for batch in src.iter_batches(batch_size=ROW_GROUP_SIZE, use_threads=False):
                # rows are copied in order, so hphigh stays sorted iff it was
                hphigh = batch["hphigh"].to_numpy()
                if (np.diff(hphigh) < 0).any() or (
                    last_hphigh is not None and hphigh[0] < last_hphigh
                ):
                    raise ValueError(f"part={part_number}: hphigh is not sorted")
                last_hphigh = hphigh[-1]

                new_batch = migrate_batch(batch, c, delta, zodi_scale, nonfunc)
                writer.write_batch(new_batch, row_group_size=ROW_GROUP_SIZE)
                n_rows += len(batch)
        f.flush()
        os.fsync(f.fileno())

    # verify before replacing anything
    if n_rows != src.metadata.num_rows or pq.read_metadata(tmp_path).num_rows != n_rows:
        raise ValueError(f"part={part_number}: row count changed during migration")
    if not pq.read_schema(tmp_path).equals(schema, check_metadata=False):
        raise ValueError(f"part={part_number}: schema changed during migration")
    if not has_been_migrated(tmp_path):
        raise ValueError(f"part={part_number}: migration marker missing from new file")

    os.replace(tmp_path, path)
    return "migrated"


def main():
    args = parse()
    paths = {
        "qr2": QR2_DIR_LOCAL if args.local else QR2_DIR_POPEYE,
        "qr3": QR3_DIR_LOCAL if args.local else QR3_DIR_POPEYE,
    }

    # provenance: written to the parquet footer of the upgrade table and of every
    # partition this run migrates, so that they record what was done to them, when,
    # and with which code (commit hash) and calibration products.
    # (a dry run records nothing, so there an uncommitted script is only a warning)
    try:
        check_script_committed()
    except RuntimeError as e:
        if not args.dry_run:
            raise
        print(f"warning (ignored in a dry run): {e}")
    if args.dry_run:
        print("DRY RUN: nothing will be written")
    print("provenance (written to the footer of every file this migration produces):")
    for key, value in provenance().items():
        print(f"  {key}: {value}")

    # download files
    # (on popeye, a missing product is an error unless --download is given)
    download = args.local or args.download
    download_if_missing(FLUX_CORR, paths, download)
    download_if_missing(GAIN_MATRIX, paths, download)
    download_if_missing(DARK_R6, paths, download)
    download_if_missing(DARK_R7, paths, download)
    download_if_missing(NONFUNC_MAP, paths, download)
    download_if_missing(ZODI_SCALING, paths, download)
    download_if_missing(SPECTRAL_WCS, paths, download)

    # prepare upgrade data
    c, delta, zodi_scale, nonfunc = massage_data(paths)
    if not args.dry_run:
        write_upgrade_table(c, delta, zodi_scale)

    # load partitions
    if args.parts:
        partitions = args.parts
    else:
        with open(PART_DB_PATH, "r") as f:
            partitions = [int(x) for x in f.readlines()]

    # do the migrations
    outcomes = Counter()
    for part_num in tqdm(partitions, desc="partitions", unit="part"):
        outcome = migrate_partition(
            part_num, c, delta, zodi_scale, nonfunc, args.dry_run
        )
        outcomes[outcome] += 1

    summary = ", ".join(f"{n} {outcome}" for outcome, n in outcomes.items())
    print(f"{summary} (of {len(partitions)} partitions)")


if __name__ == "__main__":
    main()

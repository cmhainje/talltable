import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import sys

from argparse import ArgumentParser
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor
from os.path import basename
from tqdm.auto import tqdm

from talltable.paths import DATA_DIR, IMAGE_DB_PATH, WCS_DB_PATH


# WCS keys to extract
keys = [
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
] + [
    f"{coeff}_{p}_{q}"
    for coeff in ['A', 'B']
    for p in range(4)
    for q in range(4 - p)
]


def extract_WCS(todo_tuple):
    """extract WCS parameters from filepath"""
    imageid, filepath = todo_tuple
    try:
        with fits.open(filepath) as hdul:
            header = hdul["IMAGE"].header
            data = { "imageid": imageid }
            for key in keys:
                data[key] = header[key]
            return data
    except FileNotFoundError:
        print(f"warning: {_path} not found")
        return None


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('-n', '--num-workers', type=int, default=1)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    print(
        f"building WCS table using {args.num_workers} worker"
        + ("s" if args.num_workers > 1 else "")
        + (" (using local modifications)" if args.local else "")
        )


    # *** figure out what to ingest ***

    # collect all ingested files
    ingested = duckdb.sql(f"SELECT imageid, filepath FROM read_parquet('{IMAGE_DB_PATH}')").fetchall()

    # collect the known ids
    if WCS_DB_PATH.exists():
        _result = duckdb.sql(f"SELECT imageid FROM read_parquet('{WCS_DB_PATH}')").fetchall()
        known_ids = set(x[0] for x in _result)
    else:
        known_ids = set()

    # make a list of *new* imageids and filepaths to do
    todo = []

    if args.local:
        # narrow only to files which exist locally and move paths to DATA_DIR
        local_files = set(p.name for p in DATA_DIR.glob("*.fits"))

        for (_id, _path) in ingested:
            _name = basename(_path)
            if _id not in known_ids and _name in local_files:
                todo.append((_id, DATA_DIR / _name))

        del local_files

    else:
        for (_id, _path) in ingested:
            if _id not in known_ids:
                todo.append((_id, _path))

    del ingested, known_ids

    if len(todo) == 0:
        print("no work to do!")
        sys.exit(0)

    print(f"{len(todo)} files identified to ingest WCS")


    # *** collect the WCS params of those new images ***

    # loop over the images
    if args.num_workers == 1:
        new_data = list(map(extract_WCS, tqdm(todo)))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            new_data = list(ex.map(extract_WCS, tqdm(todo)))

    new_data = [ row for row in new_data if row is not None ]

    # cast to PyArrow table with columns sorted alphabetically
    new_table = pa.table({k: [row[k] for row in new_data] for k in sorted(keys + ['imageid'])})
    del new_data

    print("built pyarrow table")


    # *** write it out ***

    if WCS_DB_PATH.exists():
        print("merging into existing file...")
        old_table = pq.ParquetFile(WCS_DB_PATH).read()
        table = pa.concat_tables(old_table, new_table)
        tmp_path = WCS_DB_PATH.with_suffix('.tmp')
        pq.write_table(
            table,
            tmp_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
        )
        tmp_path.replace(WCS_DB_PATH)

    else:
        print("writing new parquet file...")
        pq.write_table(
            new_table,
            WCS_DB_PATH,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
        )
    print("done!")


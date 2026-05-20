import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from argparse import ArgumentParser
from astropy.io import fits
from os.path import basename
from tqdm.auto import tqdm

from talltable.paths import DATA_DIR, IMAGE_DB_PATH, WCS_DB_PATH


ap = ArgumentParser()
ap.add_argument('--local', action='store_true')
args = ap.parse_args()


# *** FIGURE OUT WHICH TO INGEST ***

# collect all ingested files
ingested = duckdb.sql(f"SELECT imageid, filepath FROM read_parquet('{IMAGE_DB_PATH}')").fetchall()
#      ^ list of (id, path) tuples

# collect the known ids
if WCS_DB_PATH.exists():
    _result = duckdb.sql(f"SELECT imageid FROM read_parquet('{WCS_DB_PATH}')").fetchall()
    known_ids = set(x[0] for x in _result)
else:
    known_ids = set()

# make a list of all the new imageids and filepaths to do
todo = []

if args.local:
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

print(f"{len(todo)} files identified")


# *** collect the WCS params of those new images ***

# set up the data dict
new_data = {
    "imageid": [],
    "CRPIX1": [],
    "CRPIX2": [],
    "CRVAL1": [],
    "CRVAL2": [],
    "PC1_1": [],
    "PC1_2": [],
    "PC2_1": [],
    "PC2_2": [],
}

for coeff in ['A', 'B']:
    for i in range(4):
        for j in range(4 - i):
            col = f"{coeff}_{i}_{j}"
            new_data[col] = []

# loop over the images
for (_id, _path) in tqdm(todo):
    try:
        with fits.open(_path) as hdul:
            new_data['imageid'].append(_id)
            header = hdul["IMAGE"].header

            for key in new_data.keys():
                if key == 'imageid':
                    continue
                new_data[key].append(header.get(key, 0.0))
    except FileNotFoundError:
        print(f"warning: {_path} not found")

# cast to PyArrow table with columns sorted alphabetically
new_table = pa.table({k: new_data[k] for k in sorted(new_data)})


# *** write it out ***

if WCS_DB_PATH.exists():
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
    pq.write_table(
        new_table,
        WCS_DB_PATH,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
    )



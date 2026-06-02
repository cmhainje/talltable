import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from tqdm import tqdm

from talltable.paths import IMAGE_DB_PATH, require_env

DATA_DIR = require_env("TALLTABLE_DATA_DIR")

data = {
    "imageid": [],
    "filepath": [],
    "obsid": [],
    "t_beg": [],
    "t_end": [],
}

files = sorted(glob( str(DATA_DIR / "60907/*.fits") ))
for filepath in tqdm(files):
    with fits.open(filepath) as hdul:
        data["filepath"].append(filepath)
        data["imageid"].append(hdul["IMAGE"].header["EXPIDN"])
        data["obsid"].append(hdul["IMAGE"].header["OBSID"])
        data["t_beg"].append(hdul["IMAGE"].header["MJD-BEG"])
        data["t_end"].append(hdul["IMAGE"].header["MJD-END"])

pq.write_table(pa.table(data), IMAGE_DB_PATH)

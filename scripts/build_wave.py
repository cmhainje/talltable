import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from tqdm import tqdm

from talltable.paths import DATA_DIR, DB_DIR, WAVES_DB_PATH
from talltable.waveid import rowcoldet_to_waveid
from talltable.constants import ALL_ROW, ALL_COL
from talltable.util import byteswap


spec_dir = DATA_DIR / "spectral_wcs/cal-wcs-v4-2025-254"
print(f"looking in {spec_dir}")
spec_files = sorted(glob(str(spec_dir / "*/spectral_wcs_*.fits")))
print(f"found {len(spec_files)} spectral wcs files")

data = {
    "waveid": [],
    "wavelength": [],
    "bandwidth": [],
}

waveids_nodet = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)
for filepath in tqdm(spec_files):
    with fits.open(filepath) as f:
        det = f["CWAVE"].header["DETECTOR"]
        data["waveid"].append(waveids_nodet + (det << 24))
        data["wavelength"].append(byteswap(f["CWAVE"].data[ALL_ROW, ALL_COL]))
        data["bandwidth"].append(byteswap(f["CBAND"].data[ALL_ROW, ALL_COL]))

for k, v in data.items():
    data[k] = np.concatenate(v).squeeze()

DB_DIR.mkdir(exist_ok=True, parents=True)
pq.write_table(pa.table(data), WAVES_DB_PATH)

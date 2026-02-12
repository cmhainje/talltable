import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from tqdm import tqdm

from talltable.paths import DATA_DIR, WAVES_DB_PATH
from talltable.waveid import rowcoldet_to_waveid


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))
ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)


data = {
    "waveid": [],
    "wavelength": [],
    "bandwidth": [],
}

def byteswap(X):
    return X.view(X.dtype.newbyteorder()).byteswap()

spec_dir = DATA_DIR / "spectral_wcs/cal-wcs-v4-2025-254"
print(f"looking in {spec_dir}")
spec_files = sorted(glob(str(spec_dir / "*/spectral_wcs_*.fits")))
print(f"found {len(spec_files)} spectral wcs files")
for filepath in tqdm(spec_files):
    with fits.open(filepath) as f:
        det = f["CWAVE"].header["DETECTOR"]
        data["waveid"].append(ALL_WAVEID + (det << 24))
        data["wavelength"].append(byteswap(f["CWAVE"].data[ALL_ROW, ALL_COL]))
        data["bandwidth"].append( byteswap(f["CBAND"].data[ALL_ROW, ALL_COL]))

for k, v in data.items():
    data[k] = np.concat(v).squeeze()

pq.write_table(pa.table(data), WAVES_DB_PATH)


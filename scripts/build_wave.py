import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from tqdm import tqdm

from talltable.paths import DATA_DIR, WAVES_DB_PATH
from talltable.pixid import rowcoldet_to_pixid


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))
ALL_PIXID = rowcoldet_to_pixid(ALL_ROW, ALL_COL, 0)


data = {
    "pixid": [],
    "wavelength": [],
    "bandwidth": [],
}

def byteswap(X):
    return X.view(X.dtype.newbyteorder()).byteswap()

spec_dir = DATA_DIR / "spectral_wcs"
spec_files = sorted(glob(str(spec_dir / "spectral_wcs_*.fits")))
for filepath in tqdm(spec_files):
    with fits.open(filepath) as f:
        det = f["CWAVE"].header["DETECTOR"]
        data["pixid"].append(ALL_PIXID + (det << 24))
        data["wavelength"].append(byteswap(f["CWAVE"].data[ALL_ROW, ALL_COL]))
        data["bandwidth"].append( byteswap(f["CBAND"].data[ALL_ROW, ALL_COL]))

for k, v in data.items():
    data[k] = np.concat(v).squeeze()

pq.write_table(pa.table(data), WAVES_DB_PATH)




# data = {
#     "imageid": [],
#     "filepath": [],
#     "obsid": [],
#     "t_beg": [],
#     "t_end": [],
# }
# 
# files = sorted(glob( str(DATA_DIR / "60907/*.fits") ))
# for filepath in tqdm(files):
#     with fits.open(filepath) as hdul:
#         data["filepath"].append(filepath)
#         data["imageid"].append(hdul["IMAGE"].header["EXPIDN"])
#         data["obsid"].append(hdul["IMAGE"].header["OBSID"])
#         data["t_beg"].append(hdul["IMAGE"].header["MJD-BEG"])
#         data["t_end"].append(hdul["IMAGE"].header["MJD-END"])
# 
# pq.write_table(pa.table(data), IMAGE_DB_PATH)


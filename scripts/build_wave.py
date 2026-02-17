import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from tqdm import tqdm

from talltable.paths import DATA_DIR, WAVES_DB_PATH
from talltable.waveid import rowcoldet_to_waveid
from talltable.constants import N_WAVEPART, ALL_ROW, ALL_COL
from talltable.util import byteswap


spec_dir = DATA_DIR / "spectral_wcs/cal-wcs-v4-2025-254"
print(f"looking in {spec_dir}")
spec_files = sorted(glob(str(spec_dir / "*/spectral_wcs_*.fits")))
print(f"found {len(spec_files)} spectral wcs files")

data = {
    "waveid": [],
    "wavelength": [],
    "bandwidth": [],
    "wavepart": [],
}

waveids_nodet = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)
for filepath in tqdm(spec_files):
    with fits.open(filepath) as f:
        det = f["CWAVE"].header["DETECTOR"]
        data["waveid"].append(waveids_nodet + (det << 24))
        data["wavelength"].append(byteswap(f["CWAVE"].data[ALL_ROW, ALL_COL]))
        data["bandwidth"].append(byteswap(f["CBAND"].data[ALL_ROW, ALL_COL]))

for k, v in data.items():
    if k != "wavepart":
        data[k] = np.concatenate(v).squeeze()


def partition_edges(wavelengths, n_part):
    wavesrt = np.sort(wavelengths)
    stride = len(wavesrt) // n_part

    bounds = np.arange(1, n_part) * stride
    edges = np.zeros(n_part + 1)
    edges[1:-1] = 0.5 * (wavesrt[bounds - 1] + wavesrt[bounds])
    edges[0] = wavesrt[0] - 0.5 * (wavesrt[1] - wavesrt[0])
    edges[-1] = wavesrt[-1] + 0.5 * (wavesrt[-1] - wavesrt[-2])
    return edges


edges = partition_edges(data["wavelength"], N_WAVEPART)
data["wavepart"] = np.digitize(data["wavelength"], edges) - 1
assert np.count_nonzero(data["wavepart"] < 0) == 0
assert np.count_nonzero(data["wavepart"] >= N_WAVEPART) == 0

pq.write_table(pa.table(data), WAVES_DB_PATH)

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from glob import glob
from pathlib import Path
from tqdm import tqdm

from talltable.constants import ALL_ROW, ALL_COL
from talltable.paths import WAVES_DB_PATH, require_env
from talltable_pipeline.util import byteswap
from talltable.waveid import rowcoldet_to_waveid

# the R7 (QR3 / DR1) spectral WCS, relative to TALLTABLE_DATA_DIR_QR3. (QR2 used cal-wcs-v4-2025-254: it is
# identical except CWAVE is +6.40 nm in D5 and D6, from the on-sky recalibration of bands 5-6.)
SPECTRAL_WCS = "spectral_wcs/cal-swcs-v5-2026-191"


def parse():
    ap = argparse.ArgumentParser(
        description="build waves.parquet (wavelength and bandwidth of every pixel) from the spectral WCS product"
    )
    ap.add_argument(
        "--spectral-wcs-dir",
        type=Path,
        help=f"the spectral WCS product directory (default: {SPECTRAL_WCS} in $TALLTABLE_DATA_DIR_QR3)",
    )
    ap.add_argument("--out", type=Path, default=WAVES_DB_PATH, help=f"output file (default: {WAVES_DB_PATH})")
    return ap.parse_args()


args = parse()

spec_dir = args.spectral_wcs_dir or require_env("TALLTABLE_DATA_DIR_QR3") / SPECTRAL_WCS
print(f"looking in {spec_dir}")
spec_files = sorted(glob(str(spec_dir / "*/spectral_wcs_*.fits")))
print(f"found {len(spec_files)} spectral wcs files")
if len(spec_files) != 6:
    raise SystemExit("expected one spectral wcs file per detector (6)")

data = {
    "waveid": [],
    "wavelength": [],
    "bandwidth": [],
}

waveids_nodet = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)
detectors = []
for filepath in tqdm(spec_files):
    with fits.open(filepath) as f:
        det = int(f["CWAVE"].header["DETECTOR"])  # an int in the v4 files, a string ('1'..'6') in v5
        detectors.append(det)
        data["waveid"].append(waveids_nodet + (det << 24))
        data["wavelength"].append(byteswap(f["CWAVE"].data[ALL_ROW, ALL_COL]))
        data["bandwidth"].append(byteswap(f["CBAND"].data[ALL_ROW, ALL_COL]))
if sorted(detectors) != [1, 2, 3, 4, 5, 6]:
    raise SystemExit(f"expected detectors 1-6 exactly once, got {detectors}")

for k, v in data.items():
    data[k] = np.concatenate(v).squeeze()

# record which product this was built from, and write atomically (readers never see a half-written file)
table = pa.table(data).replace_schema_metadata({"spectral_wcs": spec_dir.name})
args.out.parent.mkdir(exist_ok=True, parents=True)
tmp_out = args.out.with_name(args.out.name + ".tmp")
pq.write_table(table, tmp_out)
os.replace(tmp_out, args.out)
print(f"wrote {args.out}")

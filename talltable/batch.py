import h5py
import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from astropy.wcs import WCS
from itertools import product
from pathlib import Path

from .constants import ALL_ROW, ALL_COL, HP_HIGH_LEVEL
from .paths import PIXEL_DB_PATH, IMAGE_PARTS_DIR, image_part_path
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat, byteswap
from .partition import partition


ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)
PARTITION_COLUMNS = ["hppart", "dfpart"]


class BatchWriter:
    def __init__(self, chunk_size=16, auto_write=True, task_id=0):
        self.chunk_size = chunk_size
        self.auto_write = auto_write
        self.task_id = task_id

        self.images = {
            "imageid": [],
            "filepath": [],
            "obsid": [],
            "t_beg": [],
            "t_end": [],
        }

        self.pixel_parts = {}

    def process_image(self, filepath):
        pixels = dict()

        def record(k, v):
            pixels[k] = v

        try:
            with fits.open(filepath) as hdul:
                idx = (ALL_ROW, ALL_COL)
                _waveid = ALL_WAVEID

                det = hdul["IMAGE"].header["DETECTOR"]
                record("waveid", _waveid + (det << 24))

                def _numpify(key, dtype=np.float32):
                    return byteswap(hdul[key].data[*idx]).astype(dtype)

                record("flux", _numpify("IMAGE"))
                record("variance", _numpify("VARIANCE"))
                record("zodi", _numpify("ZODI"))
                record("flags", _numpify("FLAGS"), dtype=np.int32)

                # sky position and derived quantities
                wcs = WCS(header=hdul["IMAGE"].header)
                ra, dec = wcs.all_pix2world(idx[1], idx[0], 0)

                hphi = hp.ang2pix(2**HP_HIGH_LEVEL, ra, dec, nest=True, lonlat=True)
                record("hphigh", hphi)

                hppart, dfpart = partition(ra, dec)
                record("hppart", hppart)
                record("dfpart", dfpart)

                # image-level stuff
                t_beg = hdul["IMAGE"].header["MJD-BEG"]
                t_end = hdul["IMAGE"].header["MJD-END"]
                obsid = hdul["IMAGE"].header["OBSID"]
                imageid = hdul["IMAGE"].header["EXPIDN"]
                record("imageid", np.array([imageid for _ in range(len(idx[0]))]))

                self.images["imageid"].append(imageid)
                self.images["filepath"].append(filepath)
                self.images["obsid"].append(obsid)
                self.images["t_beg"].append(t_beg)
                self.images["t_end"].append(t_end)

                _parts = product(*[np.unique(pixels[c]) for c in PARTITION_COLUMNS])
                for _part in _parts:
                    mask = np.logical_and.reduce(
                        [pixels[c] == v for c, v in zip(PARTITION_COLUMNS, _part)]
                    )
                    if np.count_nonzero(mask) == 0:
                        continue

                    if _part in self.pixel_parts:
                        for k, v in pixels.items():
                            self.pixel_parts[_part][k].append(v[mask])
                    else:
                        self.pixel_parts[_part] = {}
                        for k, v in pixels.items():
                            self.pixel_parts[_part][k] = [v[mask]]

        except OSError as err:
            print(f"ERROR OPENING {filepath}, {err}")
            return

        if self.auto_write:
            if self.count() >= self.chunk_size:
                self.write()

    def count(self):
        return len(self.images["filepath"])

    def clear(self):
        for key in self.images.keys():
            self.images[key] = []
        self.pixel_parts = {}

    def _write_pixels(self):
        time = f"{now_simpleformat()}_t{self.task_id}"

        for p, data in self.pixel_parts.items():
            part_path = "/".join(f"{c}={v}" for c, v in zip(PARTITION_COLUMNS, p))
            part_dir = PIXEL_DB_PATH / part_path
            part_dir.mkdir(exist_ok=True, parents=True)

            path = part_dir / f"chunk_{time}.hdf5"
            with h5py.File(path, "w") as f:
                for k, arr_list in data.items():
                    if k in PARTITION_COLUMNS:
                        continue
                    f[k] = np.concatenate(arr_list)

    def _write_images(self):
        IMAGE_PARTS_DIR.mkdir(exist_ok=True)
        db_path = image_part_path(self.task_id)

        if not db_path.exists():
            pq.write_table(pa.table(self.images), db_path)
            return

        tmp_file = Path(str(db_path) + ".tmp")
        existing_file = pq.ParquetFile(db_path)
        with pq.ParquetWriter(tmp_file, existing_file.schema_arrow) as w:
            for i in range(existing_file.num_row_groups):
                w.write_table(existing_file.read_row_group(i))
            w.write_table(pa.table(self.images))
        tmp_file.replace(db_path)

    def write(self):
        with defer_interrupt():
            self._write_pixels()
            self._write_images()
            self.clear()

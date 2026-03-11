import h5py
import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path

from .constants import ALL_ROW, ALL_COL, HP_HIGH_LEVEL, PART_MAX_LEVEL
from .paths import PIXEL_DB_PATH, IMAGE_PARTS_DIR, image_part_path, PART_DB_PATH
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat, byteswap
from .partition import level_index_to_part


ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)


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

        self.partitions = set()
        if PART_DB_PATH.exists():
            with open(PART_DB_PATH, "r") as f:
                self.partitions.update(int(p.strip()) for p in f.readlines())

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
                record("flags", _numpify("FLAGS", dtype=np.int32))

                # sky position and derived quantities
                wcs = WCS(header=hdul["IMAGE"].header)
                ra, dec = wcs.all_pix2world(idx[1], idx[0], 0)

                hphi = hp.ang2pix(2**HP_HIGH_LEVEL, ra, dec, nest=True, lonlat=True)
                record("hphigh", hphi)

                max_part = level_index_to_part(
                    PART_MAX_LEVEL, hphi >> (2 * (HP_HIGH_LEVEL - PART_MAX_LEVEL))
                )
                record("part", max_part)

                # image-level stuff
                t_beg = hdul["IMAGE"].header["MJD-BEG"]
                t_end = hdul["IMAGE"].header["MJD-END"]
                obsid = hdul["IMAGE"].header["OBSID"]
                imageid = hdul["IMAGE"].header["EXPIDN"]
                record("imageid", np.full(len(idx[0]), imageid))

                self.images["imageid"].append(imageid)
                self.images["filepath"].append(filepath)
                self.images["obsid"].append(obsid)
                self.images["t_beg"].append(t_beg)
                self.images["t_end"].append(t_end)

                u_parts, inverse = np.unique(pixels["part"], return_inverse=True)
                for i, part in enumerate(u_parts):
                    mask = inverse == i
                    if np.count_nonzero(mask) == 0:  # should not happen
                        continue

                    _part = part
                    for i in range(4):
                        if _part in self.partitions:
                            break
                        _part = _part >> 2

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

        for part, data in self.pixel_parts.items():
            print(f"writing part {part}")
            part_dir = PIXEL_DB_PATH / f"part={part}"
            part_dir.mkdir(exist_ok=True, parents=True)

            path = part_dir / f"chunk_{time}.hdf5"
            with h5py.File(path, "w") as f:
                for k, arr_list in data.items():
                    if k == "part":
                        continue
                    f[k] = np.concatenate(arr_list)

    def _write_images(self):
        IMAGE_PARTS_DIR.mkdir(exist_ok=True)
        db_path = image_part_path(self.task_id)

        if not db_path.exists():
            pq.write_table(pa.table(self.images), db_path)
            return

        # @TODO: replace with something row-oriented
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

import numpy as np
import h5py

import pyarrow as pa
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from pathlib import Path

from .paths import PIXEL_DB_PATH, IMAGE_PARTS_DIR, image_part_path
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))
ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)

HP_LO_LEVEL = 8
HP_HI_LEVEL = 24

HEALPIX_LO = HEALPix(nside=2**HP_LO_LEVEL, order="nested", frame="icrs")
HEALPIX_HI = HEALPix(nside=2**HP_HI_LEVEL, order="nested", frame="icrs")


class BatchWriter:
    def __init__(self, chunk_size=16, skip_bad=True, auto_write=True, task_id=0):
        self.chunk_size = chunk_size
        self.skip_bad = skip_bad
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
        def byteswap(X):
            return X.view(X.dtype.newbyteorder()).byteswap()

        pixels = dict()

        def record(k, v):
            pixels[k] = v

        try:
            with fits.open(filepath) as hdul:
                if self.skip_bad:
                    flags = hdul["FLAGS"].data[ALL_ROW, ALL_COL]
                    good = flags & ~(1 << 21) == 0
                    if np.count_nonzero(good) == 0:
                        return None
                    idx = (ALL_ROW[good], ALL_COL[good])
                    _waveid = ALL_WAVEID[good]
                else:
                    idx = (ALL_ROW, ALL_COL)
                    _waveid = ALL_WAVEID[good]

                det = hdul["IMAGE"].header["DETECTOR"]
                record("waveid",    _waveid + (det << 24))

                record("flux",     byteswap(hdul["IMAGE"].data[*idx]).astype(np.float32))
                record("variance", byteswap(hdul["VARIANCE"].data[*idx]).astype(np.float32))
                record("zodi",     byteswap(hdul["ZODI"].data[*idx]).astype(np.float32))

                if self.skip_bad:
                    record("known", hdul["FLAGS"].data[*idx] & (1 << 21) == 1)
                else:
                    record("flags", byteswap(hdul["FLAGS"].data[*idx]).astype(np.int32))

                # sky position and derived quantities
                wcs = WCS(header=hdul["IMAGE"].header)
                ra, dec = wcs.wcs_pix2world(idx[1], idx[0], 0)
                sc = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
                record("hphigh", HEALPIX_HI.skycoord_to_healpix(sc))
                # record("hppart", HEALPIX_LO.skycoord_to_healpix(sc))
                record("hppart", pixels["hphigh"] >> (2 * (HP_HI_LEVEL - HP_LO_LEVEL)))

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

                _parts = np.unique(pixels["hppart"])
                for _p in _parts:
                    mask = pixels["hppart"] == _p
                    if _p in self.pixel_parts:
                        for k, v in pixels.items():
                            self.pixel_parts[_p][k].append(v[mask])
                    else:
                        self.pixel_parts[_p] = {}
                        for k, v in pixels.items():
                            self.pixel_parts[_p][k] = [v[mask]]

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
            part_dir = PIXEL_DB_PATH / f"hppart={p}"
            part_dir.mkdir(exist_ok=True)

            path = part_dir / f"chunk_{time}.hdf5"
            with h5py.File(path, 'w') as f:
                for k, arr_list in data.items():
                    if k == 'hppart':
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

import numpy as np

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from datetime import datetime
from pathlib import Path

from .paths import IMAGE_DB_PATH, PIXEL_DB_PATH
from .pixid import rowcoldet_to_pixid
from .util import defer_interrupt


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))

HEALPIX02 = HEALPix(nside=2**2, order="nested", frame="icrs")
HEALPIX20 = HEALPix(nside=2**20, order="nested", frame="icrs")


def now_simpleformat() -> str:
    time = datetime.now().isoformat(timespec='seconds')
    time = time.replace('-', '')
    time = time.replace(':', '')
    time = time.replace('T', '-')
    return time


class BatchWriter:
    def __init__(self, chunk_size=16, skip_bad=True, auto_write=True):
        self.chunk_size = chunk_size
        self.skip_bad = skip_bad
        self.auto_write = auto_write

        self.file_opt = ds.ParquetFileFormat().make_write_options(
            compression="zstd",
            compression_level=3,
        )

        self.images = {
            "imageid": [],
            "filepath": [],
            "obsid": [],
            "t_beg": [],
            "t_end": [],
        }

        self.pixel_tables = []

        # self.pixels = {
        #     # "row": [],
        #     # "col": [],
        #     "pixid": [],
        #     # "ra": [],
        #     # "dec": [],
        #     # "wavelen": [],
        #     # "waveband": [],
        #     "ux": [],
        #     "uy": [],
        #     "uz": [],
        #     "flux": [],
        #     "variance": [],
        #     "zodi": [],
        #     # "flags": [],
        #     # "known": [],
        #     "hp02": [],
        #     "hp20": [],
        #     "imageid": [],
        # }

        # if self.skip_bad:
        #     self.pixels["known"] = []
        # else:
        #     self.pixels["flags"] = []

    def process_image(self, filepath):
        def byteswap(X):
            return X.view(X.dtype.newbyteorder()).byteswap()

        pixels = dict()

        def record(k, v):
            pixels[k] = v
            # self.pixels[k].append(v)

        try:
            with fits.open(filepath) as hdul:
                if self.skip_bad:
                    flags = hdul["FLAGS"].data[ALL_ROW, ALL_COL]
                    good = flags & ~(1 << 21) == 0
                    if np.count_nonzero(good) == 0:
                        return None
                    idx = (ALL_ROW[good], ALL_COL[good])
                else:
                    idx = (ALL_ROW, ALL_COL)

                # record("row", idx[0].astype(np.int32))
                # record("col", idx[1].astype(np.int32))
                det = hdul["IMAGE"].header["DETECTOR"]
                record("pixid",    rowcoldet_to_pixid(*idx, det))

                record("flux",     byteswap(hdul["IMAGE"].data[*idx]).astype(np.float32))
                record("variance", byteswap(hdul["VARIANCE"].data[*idx]).astype(np.float32))
                record("zodi",     byteswap(hdul["ZODI"].data[*idx]).astype(np.float32))

                if self.skip_bad:
                    record("known", hdul["FLAGS"].data[*idx] & (1 << 21) == 1)
                else:
                    record("flags", byteswap(hdul["FLAGS"].data[*idx]).astype(np.int32))

                # sky position and derived quantities
                wcs = WCS(header=hdul["IMAGE"].header)
                ra, dec = wcs.wcs_pix2world(*idx, 0)
                # record("ra", ra)
                # record("dec", dec)

                record("ux", np.cos(np.radians(dec)) * np.cos(np.radians(ra)))
                record("uy", np.cos(np.radians(dec)) * np.sin(np.radians(ra)))
                record("uz", np.sin(np.radians(dec)))

                sc = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
                record("hp02", HEALPIX02.skycoord_to_healpix(sc))
                record("hp20", HEALPIX20.skycoord_to_healpix(sc))

                # wavelength
                # del hdul["IMAGE"].header["A_ORDER"]
                # del hdul["IMAGE"].header["B_ORDER"]
                # del hdul["IMAGE"].header["AP_ORDER"]
                # del hdul["IMAGE"].header["BP_ORDER"]
                # wave_wcs = WCS(header=hdul["IMAGE"].header, fobj=hdul, key="W")
                # wave_wcs.sip = None
                # lam, dlam = wave_wcs.wcs_pix2world(idx, 0).T
                # record("wavelen", lam)
                # record("waveband", dlam)

                # image-level stuff
                t_beg = hdul["IMAGE"].header["MJD-BEG"]
                t_end = hdul["IMAGE"].header["MJD-END"]
                obsid = hdul["IMAGE"].header["OBSID"]
                imageid = hdul["IMAGE"].header["EXPIDN"]
                record("imageid", np.array([imageid for _ in range(len(idx[0]))]))

                self.pixel_tables.append(pa.table(pixels))

                self.images["imageid"].append(imageid)
                self.images["filepath"].append(filepath)
                self.images["obsid"].append(obsid)
                self.images["t_beg"].append(t_beg)
                self.images["t_end"].append(t_end)
        except OSError:
            print(f"ERROR OPENING {filepath}")
            return

        if self.auto_write:
            if self.count() >= self.chunk_size:
                self.write()

    def count(self):
        # return len(self.images["imageid"])
        return len(self.pixel_tables)

    def clear(self):
        for key in self.images.keys():
            self.images[key] = []
        self.pixel_tables = []
        # for key in self.pixels.keys():
        #     self.pixels[key] = []

    # def flatten_pixels(self):
    #     for key, value in self.pixels.items():
    #         self.pixels[key] = np.concatenate(value)

    def _write_pixels(self):
        # self.flatten_pixels()
        time = now_simpleformat()
        ds.write_dataset(
            data=pa.concat_tables(self.pixel_tables),
            base_dir=PIXEL_DB_PATH,
            partitioning=["hp02"],
            partitioning_flavor="hive",
            format="parquet",
            file_options=self.file_opt,
            basename_template=f"release_{time}_{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )

    def _write_images(self):
        if not IMAGE_DB_PATH.exists():
            pq.write_table(pa.table(self.images), IMAGE_DB_PATH)
            return

        tmp_file = Path(str(IMAGE_DB_PATH) + ".tmp")
        existing_file = pq.ParquetFile(IMAGE_DB_PATH)
        with pq.ParquetWriter(tmp_file, existing_file.schema_arrow) as w:
            for i in range(existing_file.num_row_groups):
                w.write_table(existing_file.read_row_group(i))
            w.write_table(pa.table(self.images))
        tmp_file.replace(IMAGE_DB_PATH)  # overwrite existing file

    def write(self):
        with defer_interrupt():
            self._write_pixels()
            self._write_images()
            self.clear()

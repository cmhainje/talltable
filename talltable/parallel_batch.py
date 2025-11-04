import numpy as np

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .paths import IMAGE_DB_PATH, PIXEL_DB_PATH
from .pixid import rowcoldet_to_pixid
from .util import defer_interrupt


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))

HEALPIX02 = HEALPix(nside=2**2, order="nested", frame="icrs")
HEALPIX20 = HEALPix(nside=2**20, order="nested", frame="icrs")


SKIP_BAD = True


def process_image(filepath):
    def byteswap(X):
        return X.view(X.dtype.newbyteorder()).byteswap()

    images = dict()
    pixels = dict()

    def record(k, v):
        pixels[k] = v

    try:
        with fits.open(filepath) as hdul:
            if SKIP_BAD:
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

            if SKIP_BAD:
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

            pixels = pa.table(pixels)

            images["imageid"] = imageid
            images["filepath"] = filepath
            images["obsid"] = obsid
            images["t_beg"] = t_beg
            images["t_end"] = t_end

        return images, pixels
    except OSError:
        print(f"ERROR OPENING {filepath}")
        return



class ParallelBatchWriter:
    def __init__(self, num_workers=8):
        self.num_workers = num_workers

        self.file_opt = ds.ParquetFileFormat().make_write_options(
            compression="zstd",
            compression_level=3,
        )

    def process_batch(self, filepaths):
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            images, pixels = list(zip(*list( r for r in ex.map(process_image, filepaths) if r is not None )))
        images = pa.table({k: [row[k] for row in images] for k in images[0]})

        with defer_interrupt():
            self.write_images(images)
            self.write_pixels(pa.concat_tables(pixels))

    def write_pixels(self, pixels):
        ds.write_dataset(
            data=pa.table(pixels),
            base_dir=PIXEL_DB_PATH,
            partitioning=["hp02"],
            partitioning_flavor="hive",
            format="parquet",
            file_options=self.file_opt,
            basename_template="release_{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )

    def write_images(self, images):
        db_path = IMAGE_DB_PATH

        if not db_path.exists():
            pq.write_table(images, db_path)
            return

        tmp_file = Path(str(db_path) + ".tmp")
        existing_file = pq.ParquetFile(db_path)
        with pq.ParquetWriter(tmp_file, existing_file.schema_arrow) as w:
            for i in range(existing_file.num_row_groups):
                w.write_table(existing_file.read_row_group(i))
            w.write_table(pa.table(images))
        tmp_file.replace(db_path)  # overwrite existing file

import numpy as np
import h5py

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
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat


_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing='ij'))
ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)

HP_LO_LEVEL = 8
HP_HI_LEVEL = 24

HEALPIX_LO = HEALPix(nside=2**HP_LO_LEVEL, order="nested", frame="icrs")
HEALPIX_HI = HEALPix(nside=2**HP_HI_LEVEL, order="nested", frame="icrs")


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
                _waveid = ALL_WAVEID[good]
            else:
                idx = (ALL_ROW, ALL_COL)
                _waveid = ALL_WAVEID[good]

            det = hdul["IMAGE"].header["DETECTOR"]
            record("waveid",    _waveid + (det << 24))

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
            sc = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            record("hphigh", HEALPIX_HI.skycoord_to_healpix(sc))
            record("hppart", pixels["hphigh"] >> (2 * (HP_HI_LEVEL - HP_LO_LEVEL)))

            # image-level stuff
            t_beg = hdul["IMAGE"].header["MJD-BEG"]
            t_end = hdul["IMAGE"].header["MJD-END"]
            obsid = hdul["IMAGE"].header["OBSID"]
            imageid = hdul["IMAGE"].header["EXPIDN"]
            record("imageid", np.array([imageid for _ in range(len(idx[0]))]))

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

        self.time = now_simpleformat()

        self.file_opt = ds.ParquetFileFormat().make_write_options(
            compression="zstd",
            compression_level=3,
        )

        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)

    def __del__(self):
        self.executor.shutdown(wait=True)

    def process_batch(self, filepaths):
        images, pixels = list(zip(*[r for r in self.executor.map(process_image, filepaths) if r is not None]))
        images = pa.table({k: [row[k] for row in images] for k in images[0]})
        pixels = {k: np.concatenate([d[k] for d in pixels]) for k in pixels[0]}
        pixel_parts = { p: { k: v[ pixels["hppart"] == p ] for k, v in pixels.items() if k != "hppart" } for p in np.unique(pixels["hppart"]) }

        with defer_interrupt():
            self.write_images(images)
            self.write_pixels(pixel_parts)

    def write_pixels(self, pixel_parts):
        # time = now_simpleformat()

        def _append(f, k, arr):
            if k in f:
                f[k].resize(len(f[k]) + len(arr), axis=0)
                f[k][-len(arr):] = arr
            else:
                f.create_dataset(k, data=arr, maxshape=(None,))

        for p, data in pixel_parts.items():
            part_dir = PIXEL_DB_PATH / f"hppart={p}"
            part_dir.mkdir(exist_ok=True)

            path = part_dir / f"chunk_{self.time}.hdf5"
            with h5py.File(path, 'a') as f:
                for k, arr in data.items():
                    if k != 'hppart':
                        _append(f, k, arr)


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

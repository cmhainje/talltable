import ctypes
import h5py
import healpy as hp
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from astropy.wcs import WCS
from dataclasses import dataclass
from pathlib import Path

try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass

from .constants import ALL_ROW, ALL_COL, HP_HIGH_LEVEL, PART_MAX_LEVEL, PART_MIN_LEVEL
from .paths import PIXEL_DB_PATH, IMAGE_PARTS_DIR, image_part_path, PART_DB_PATH
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat, byteswap
from .partition import level_index_to_part


logger = logging.getLogger(__name__)

NPIX = len(ALL_ROW)
ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)

PIXEL_DTYPES = {
    "waveid": np.int32,
    "flux": np.float32,
    "variance": np.float32,
    "zodi": np.float32,
    "flags": np.int32,
    "hphigh": np.int64,
    "imageid": np.int64,
    "part": np.int32,
}
PIXEL_COLUMNS = list(PIXEL_DTYPES.keys())


@dataclass
class FITSData:
    filepath: str
    image: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    flags: np.ndarray
    header: fits.Header


def read_image(filepath):
    """Read FITS file and return raw data. Pure I/O, safe to call from a thread."""
    with fits.open(filepath) as hdul:
        idx = (ALL_ROW, ALL_COL)
        return FITSData(
            filepath=filepath,
            image=hdul["IMAGE"].data[*idx].copy(),
            variance=hdul["VARIANCE"].data[*idx].copy(),
            zodi=hdul["ZODI"].data[*idx].copy(),
            flags=hdul["FLAGS"].data[*idx].copy(),
            header=hdul["IMAGE"].header.copy(),
        )


class BatchWriter:
    def __init__(self, chunk_size=32, auto_write=True, task_id=0):
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

        # pre-allocate pixel buffer: one flat array per column, reused across chunks
        self._pixel_buf = {
            k: np.empty(chunk_size * NPIX, dtype=dt)
            for k, dt in PIXEL_DTYPES.items()
        }
        self._buf_pos = 0  # number of images currently buffered

    def process_image(self, data):
        if isinstance(data, str):
            try:
                data = read_image(data)
            except OSError as err:
                logger.error("error opening %s: %s", data, err)
                return

        det = data.header["DETECTOR"]
        waveid = ALL_WAVEID + (det << 24)

        flux = byteswap(data.image).astype(np.float32)
        variance = byteswap(data.variance).astype(np.float32)
        zodi = byteswap(data.zodi).astype(np.float32)
        flags = byteswap(data.flags).astype(np.int32)

        wcs = WCS(header=data.header)
        ra, dec = wcs.all_pix2world(ALL_COL, ALL_ROW, 0)

        hphi = hp.ang2pix(2**HP_HIGH_LEVEL, ra, dec, nest=True, lonlat=True)

        max_part = level_index_to_part(
            PART_MAX_LEVEL, hphi >> (2 * (HP_HIGH_LEVEL - PART_MAX_LEVEL))
        )

        # resolve each pixel's partition by walking up the hierarchy
        part = max_part.copy()
        u_parts, inverse = np.unique(max_part, return_inverse=True)
        for j, part in enumerate(u_parts):
            _part = part
            for _ in range(PART_MAX_LEVEL - PART_MIN_LEVEL):
                if _part in self.partitions:
                    break
                _part = _part >> 2
            if _part != part:
                part[inverse == j] = _part

        imageid = np.full(NPIX, data.header["EXPIDN"])

        # write into pre-allocated buffer at current offset
        i = self._buf_pos
        sl = slice(i * NPIX, (i + 1) * NPIX)
        self._pixel_buf["waveid"][sl] = waveid
        self._pixel_buf["flux"][sl] = flux
        self._pixel_buf["variance"][sl] = variance
        self._pixel_buf["zodi"][sl] = zodi
        self._pixel_buf["flags"][sl] = flags
        self._pixel_buf["hphigh"][sl] = hphi
        self._pixel_buf["imageid"][sl] = imageid
        self._pixel_buf["part"][sl] = part
        self._buf_pos += 1

        # accumulate image metadata
        self.images["imageid"].append(data.header["EXPIDN"])
        self.images["filepath"].append(data.filepath)
        self.images["obsid"].append(data.header["OBSID"])
        self.images["t_beg"].append(data.header["MJD-BEG"])
        self.images["t_end"].append(data.header["MJD-END"])

        if self.auto_write and self.count() >= self.chunk_size:
            self.write()

    def count(self):
        return len(self.images["filepath"])

    def clear(self):
        for key in self.images:
            self.images[key] = []
        self._buf_pos = 0

    def _write_pixels(self):
        time = f"{now_simpleformat()}_t{self.task_id}"
        n = self._buf_pos * NPIX

        # sort order from part (view into pre-allocated buffer, no copy)
        part = self._pixel_buf["part"][:n]
        sort_idx = np.argsort(part, kind="mergesort")
        sorted_part = part[sort_idx]

        # compute partition boundary indices
        part_ids, part_starts = np.unique(sorted_part, return_index=True)
        part_ends = np.empty_like(part_starts)
        part_ends[:-1] = part_starts[1:]
        part_ends[-1] = n
        del sorted_part

        # write to a single flat HDF5 file via temp + rename
        PIXEL_DB_PATH.mkdir(exist_ok=True, parents=True)
        tmp_path = PIXEL_DB_PATH / f"chunk_{time}.hdf5.tmp"
        final_path = PIXEL_DB_PATH / f"chunk_{time}.hdf5"

        with h5py.File(tmp_path, "w") as f:
            for k in PIXEL_COLUMNS:
                if k == "part":
                    continue
                f[k] = self._pixel_buf[k][:n][sort_idx]

            f.attrs["part_ids"] = part_ids
            f.attrs["part_starts"] = part_starts
            f.attrs["part_ends"] = part_ends

        del sort_idx
        tmp_path.rename(final_path)

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
            _malloc_trim()

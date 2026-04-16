import healpy as hp
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.io import fits
from astropy.wcs import WCS
from dataclasses import dataclass
from pathlib import Path

from .constants import ALL_ROW, ALL_COL, HP_HIGH_LEVEL, PART_MAX_LEVEL, PART_MIN_LEVEL
from .paths import IMAGE_PARTS_DIR, image_part_path, PART_DB_PATH, SCRATCH_DIR
from .waveid import rowcoldet_to_waveid
from .util import defer_interrupt, now_simpleformat, byteswap
from .partition import level_index_to_part


logger = logging.getLogger(__name__)

ALL_WAVEID = rowcoldet_to_waveid(ALL_ROW, ALL_COL, 0)

PIXEL_COLUMNS = ["waveid", "flux", "variance", "zodi", "flags", "hphigh", "imageid", "hppart"]

# Column order and dtypes for the binary chunk format.
CHUNK_COLUMNS = [
    ("flux",     np.float32),
    ("variance", np.float32),
    ("zodi",     np.float32),
    ("flags",    np.int32),
    ("hphigh",   np.int64),
    ("waveid",   np.int32),
    ("imageid",  np.int64),
]


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
    def __init__(self, chunk_size=24, auto_write=True, task_id=0):
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

        self.pixel_buffer = {k: [] for k in PIXEL_COLUMNS}

    def process_image(self, data):
        if isinstance(data, str):
            try:
                data = read_image(data)
            except OSError as err:
                logger.error("error opening %s: %s", data, err)
                return

        npix = len(ALL_ROW)

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
        hppart = max_part.copy()
        u_parts, inverse = np.unique(max_part, return_inverse=True)
        for j, part in enumerate(u_parts):
            _part = part
            for _ in range(PART_MAX_LEVEL - PART_MIN_LEVEL):
                if _part in self.partitions:
                    break
                _part = _part >> 2
            if _part != part:
                hppart[inverse == j] = _part

        imageid = np.full(npix, data.header["EXPIDN"])

        # accumulate into flat buffer
        self.pixel_buffer["waveid"].append(waveid)
        self.pixel_buffer["flux"].append(flux)
        self.pixel_buffer["variance"].append(variance)
        self.pixel_buffer["zodi"].append(zodi)
        self.pixel_buffer["flags"].append(flags)
        self.pixel_buffer["hphigh"].append(hphi)
        self.pixel_buffer["imageid"].append(imageid)
        self.pixel_buffer["hppart"].append(hppart)

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
        self.pixel_buffer = {k: [] for k in PIXEL_COLUMNS}

    def _write_pixels(self):
        suffix = f"{now_simpleformat()}_t{self.task_id}"

        # concatenate all buffered arrays
        data = {}
        for k, arr_list in self.pixel_buffer.items():
            data[k] = np.concatenate(arr_list)

        # sort by hphigh; since active partitions tile the hphigh space without
        # overlap, this also makes data for the same partition contiguous, AND
        # leaves each partition's slice sorted by hphigh for efficient merging
        # during compaction.
        sort_idx = np.argsort(data["hphigh"], kind="mergesort")
        for k in data:
            data[k] = data[k][sort_idx]

        # compute partition boundary indices
        # np.unique returns values sorted by partition ID, but data is sorted
        # by hphigh, so partitions appear in hphigh order (not ID order).
        # Re-sort by occurrence position so part_starts is monotonic.
        hppart = data["hppart"]
        part_ids, part_starts = np.unique(hppart, return_index=True)
        order = np.argsort(part_starts)
        part_ids = part_ids[order]
        part_starts = part_starts[order]
        part_ends = np.empty_like(part_starts)
        part_ends[:-1] = part_starts[1:]
        part_ends[-1] = len(hppart)

        # write to a single flat binary file
        SCRATCH_DIR.mkdir(exist_ok=True, parents=True)
        path = SCRATCH_DIR / f"chunk_{suffix}.bin"

        with open(path, "wb") as f:
            # header: partition index
            np.array([len(part_ids)], dtype=np.uint32).tofile(f)
            part_ids.astype(np.uint32).tofile(f)
            part_starts.astype(np.uint64).tofile(f)
            part_ends.astype(np.uint64).tofile(f)
            np.array([len(hppart)], dtype=np.uint64).tofile(f)
            # columns
            for name, dtype in CHUNK_COLUMNS:
                data[name].astype(dtype).tofile(f)

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

import h5py
import logging
import numpy as np
import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.constants import HP_HIGH_LEVEL, MAX_ROWS_PER_PART, PART_MAX_LEVEL
from talltable.partition import part_to_level_index, level_index_to_part
from talltable.paths import PIXEL_DB_PATH


task_id = int(os.environ.get("SLURM_PROCID", 0))
num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
job_id = os.environ.get("SLURM_JOB_ID", 0)
out_file = os.environ.get("SLURM_JOB_STDOUT", f"./slurm-{job_id}.out")


logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [task {task_id}] %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def scan_chunk_files():
    """Scan flat HDF5 chunk files and build a partition -> [(file, start, end)] mapping."""
    h5_files = sorted(PIXEL_DB_PATH.glob("chunk_*.hdf5"))
    partition_index = {}

    for fpath in h5_files:
        try:
            with h5py.File(fpath, "r") as f:
                part_ids = f.attrs["part_ids"]
                part_starts = f.attrs["part_starts"]
                part_ends = f.attrs["part_ends"]
        except (OSError, KeyError) as e:
            logger.warning(f"skipping {fpath}: {e}")
            continue

        for pid, start, end in zip(part_ids, part_starts, part_ends):
            pid = int(pid)
            if pid not in partition_index:
                partition_index[pid] = []
            partition_index[pid].append((fpath, int(start), int(end)))

    logger.info(f"scanned {len(h5_files)} chunk files, found {len(partition_index)} partitions")
    return partition_index


def read_partition_data(sources):
    """Read a partition's data from flat HDF5 files using contiguous slices."""
    tables = []
    for fpath, start, end in sources:
        data = {}
        with h5py.File(fpath, "r") as f:
            for key in f.keys():
                data[key] = f[key][start:end]
        try:
            tables.append(pa.table(data))
        except pa.lib.ArrowInvalid as e:
            msg = f"failed to read slice [{start}:{end}] from {fpath}.\n"
            msg += "data dict included:\n"
            for key in data:
                if isinstance(data[key], np.ndarray):
                    msg += f"{key}: np.array [{data[key].shape}]\n"
                else:
                    msg += f"{key}: {data[key]}\n"
            msg += f"error message:\n{e}"
            raise RuntimeError(msg)
    return tables


def main():
    # scan flat HDF5 chunk files for partition boundaries
    partition_index = scan_chunk_files()

    keys = sorted(partition_index.keys())
    if num_tasks > 1:
        keys = keys[task_id::num_tasks]

    for part in (tqdm(keys, desc="task 0", unit="part") if task_id == 0 else keys):
        sources = partition_index[part]
        if len(sources) == 0:
            continue

        part_dir = PIXEL_DB_PATH / f"part={part}"
        part_dir.mkdir(exist_ok=True, parents=True)
        pq_path = part_dir / "compacted.parquet"
        staging_path = part_dir / "compacted_new.parquet"

        # read data from all sources
        tables = []

        # flat chunk files (contiguous slices)
        if sources:
            tables.extend(read_partition_data(sources))

        # include existing compacted parquet if present
        if pq_path.exists():
            try:
                tables.append(pq.ParquetFile(pq_path).read())
            except pa.lib.ArrowInvalid as e:
                msg = f"failed to open Parquet file {pq_path} with error message:\n{e}"
                raise RuntimeError(msg)

        table = pa.concat_tables(tables)

        # sort
        sort_keys = [("hphigh", "ascending")]
        table = table.sort_by(sort_keys)
        sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)

        # check if it's too big
        level, index = part_to_level_index(part)
        if len(table) > MAX_ROWS_PER_PART and level < PART_MAX_LEVEL:
            # split into 4 subpartitions at the next level
            _level = level + 1
            for k in range(4):
                _index = (index << 2) + k
                _p = level_index_to_part(_level, _index)
                _part_dir = PIXEL_DB_PATH / f"part={_p}"
                _part_dir.mkdir(exist_ok=True, parents=True)

                _lo = (_index)     << 2 * (HP_HIGH_LEVEL - _level)
                _hi = (_index + 1) << 2 * (HP_HIGH_LEVEL - _level)

                _mask = (pc.field("hphigh") >= _lo) & (pc.field("hphigh") < _hi)
                _table = table.filter(_mask)

                _pq_path = _part_dir / "compacted_new.parquet"
                pq.write_table(
                    _table,
                    _pq_path,
                    compression="zstd",
                    compression_level=3,
                    sorting_columns=sorting_cols,
                )

            # write a marker so post_compact knows to clean up the parent
            split_marker = part_dir / ".split"
            split_marker.touch()

        else:
            pq.write_table(
                table,
                staging_path,
                compression="zstd",
                compression_level=3,
                sorting_columns=sorting_cols,
            )

        del tables, table
        pa.default_memory_pool().release_unused()

        for handler in logger.handlers:
            handler.flush()


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()

import h5py
import logging
import numpy as np
import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.constants import HP_HIGH_LEVEL, MAX_ROWS_PER_PART, PART_MAX_LEVEL
from talltable.partition import part_to_level_index, level_index_to_part
from talltable.paths import PIXEL_DB_PATH, IMAGE_DB_PATH, IMAGE_PARTS_DIR


logger = logging.getLogger(__name__)
task_id = int(os.environ.get("SLURM_PROCID", 0))
num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
job_id = os.environ.get("SLURM_JOB_ID", 0)
out_file = os.environ.get("SLURM_JOB_STDOUT", f"./slurm-{job_id}.out")


def merge_image_parts():
    """merge new partial image parquet files into the final images table"""
    part_files = sorted(IMAGE_PARTS_DIR.glob("image_task*.parquet"))
    if not part_files:
        logging.info("no image part files to merge")
        return

    logging.info(f"merging {len(part_files)} image part files")
    tables = []
    if IMAGE_DB_PATH.exists():
        tables.append(pq.read_table(IMAGE_DB_PATH))
    for f in part_files:
        tables.append(pq.read_table(f))

    merged = pa.concat_tables(tables)
    tmp_file = Path(str(IMAGE_DB_PATH) + ".tmp")
    pq.write_table(merged, tmp_file)
    tmp_file.replace(IMAGE_DB_PATH)

    for f in part_files:
        f.unlink()
    logging.info(f"merged into {IMAGE_DB_PATH} ({len(merged)} rows)")


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
            logging.warning(f"skipping {fpath}: {e}")
            continue

        for pid, start, end in zip(part_ids, part_starts, part_ends):
            pid = int(pid)
            if pid not in partition_index:
                partition_index[pid] = []
            partition_index[pid].append((fpath, int(start), int(end)))

    logging.info(f"scanned {len(h5_files)} chunk files, found {len(partition_index)} partitions")
    return partition_index, h5_files


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
    logging.info(f"processing index {task_id} of {num_tasks} tasks")

    # merge image parts (only do once)
    if task_id == 0:
        merge_image_parts()

    # scan flat HDF5 chunk files for partition boundaries
    partition_index, h5_files = scan_chunk_files()

    # also pick up any old-style per-partition HDF5 files (from previous ingests)
    for p in sorted(PIXEL_DB_PATH.glob("part=*/*.hdf5")):
        part = int(p.parts[-2].split("=")[1])
        # use sentinel (start=-1, end=-1) to indicate old-style file
        if part not in partition_index:
            partition_index[part] = []
        partition_index[part].append((p, -1, -1))

    keys = sorted(partition_index.keys())
    if num_tasks > 1:
        keys = keys[task_id::num_tasks]

    for part in tqdm(keys):
        try:
            level, index = part_to_level_index(part)
            sources = partition_index[part]
            if len(sources) == 0:
                continue

            part_dir = PIXEL_DB_PATH / f"part={part}"
            part_dir.mkdir(exist_ok=True, parents=True)
            pq_path = part_dir / "compacted.parquet"
            tmp_path = part_dir / "tmp.parquet"

            # read data from all sources
            tables = []

            # flat chunk files (contiguous slices)
            flat_sources = [(f, s, e) for f, s, e in sources if s >= 0]
            if flat_sources:
                tables.extend(read_partition_data(flat_sources))

            # old-style per-partition HDF5 files
            old_h5_files = [f for f, s, _ in sources if s < 0]
            for filepath in old_h5_files:
                data = {}
                with h5py.File(filepath, "r") as f:
                    for key in f.keys():
                        data[key] = np.atleast_1d(f[key][:].squeeze())
                try:
                    tables.append(pa.table(data))
                except pa.lib.ArrowInvalid as e:
                    msg = f"failed to process h5 file {filepath}.\n"
                    msg += f"error message:\n{e}"
                    raise RuntimeError(msg)

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

            # check if its too big
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

                    _pq_path = _part_dir / "compacted.parquet"
                    pq.write_table(
                        _table,
                        _pq_path,
                        compression="zstd",
                        compression_level=3,
                        sorting_columns=sorting_cols,
                    )

                # clean up old per-partition HDF5 files
                for filepath in old_h5_files:
                    filepath.unlink()
                if pq_path.exists():
                    pq_path.unlink()
                if part_dir.exists() and not any(part_dir.iterdir()):
                    part_dir.rmdir()

            else:
                pq.write_table(
                    table,
                    tmp_path,
                    compression="zstd",
                    compression_level=3,
                    sorting_columns=sorting_cols,
                )

                # clean up old per-partition HDF5 files
                for filepath in old_h5_files:
                    filepath.unlink()
                tmp_path.replace(pq_path)

        except RuntimeError as e:
            logging.warning(f"warning: failed processing partition {part}:\n{e}\ncontinuing...")

        for handler in logger.handlers:
            handler.flush()

    # flat chunk HDF5 files can be deleted after all tasks complete
    # if h5_files:
    #     logging.info(
    #         f"compact done. to clean up {len(h5_files)} flat chunk files, run:\n"
    #         f"  rm {PIXEL_DB_PATH}/chunk_*.hdf5"
    #     )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename=out_file + f".{task_id}",
        format=f"%(asctime)s [{task_id}] %(levelname)s %(message)s",
    )
    with logging_redirect_tqdm():
        main()

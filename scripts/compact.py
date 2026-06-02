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
from talltable.paths import PIXEL_DB_PATH, SCRATCH_DIR
from talltable_pipeline.util import defer_interrupt
from talltable_pipeline.batch import CHUNK_COLUMNS


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
    """Scan binary chunk files and build a partition -> [(file, start, end)] mapping."""
    bin_files = sorted(SCRATCH_DIR.glob("chunk_*.bin"))
    partition_index = {}

    for fpath in bin_files:
        try:
            with open(fpath, "rb") as f:
                num_part = np.fromfile(f, dtype=np.uint32, count=1)[0]
                part_ids = np.fromfile(f, dtype=np.uint32, count=num_part)
                part_starts = np.fromfile(f, dtype=np.uint64, count=num_part)
                part_ends = np.fromfile(f, dtype=np.uint64, count=num_part)
        except (OSError, ValueError) as e:
            logger.warning(f"skipping {fpath}: {e}")
            continue

        for pid, start, end in zip(part_ids, part_starts, part_ends):
            pid = int(pid)
            if pid not in partition_index:
                partition_index[pid] = []
            partition_index[pid].append((fpath, int(start), int(end)))

    if task_id == 0:
        logger.info(f"scanned {len(bin_files)} chunk files, found {len(partition_index)} partitions")
    return partition_index


def read_partition_data(sources):
    """Read a partition's data from binary chunk files using contiguous slices."""
    tables = []
    for fpath, start, end in sources:
        with open(fpath, "rb") as f:
            num_part = np.fromfile(f, dtype=np.uint32, count=1)[0]
            header_size = 4 + num_part * 20 + 8
            f.seek(header_size - 8)
            num_rows = int(np.fromfile(f, dtype=np.uint64, count=1)[0])

            data = {}
            col_offset = header_size
            for name, dtype in CHUNK_COLUMNS:
                itemsize = np.dtype(dtype).itemsize
                f.seek(col_offset + start * itemsize)
                data[name] = np.fromfile(f, dtype=dtype, count=end - start)
                col_offset += num_rows * itemsize

        tables.append(pa.table({k: data[k] for k in sorted(data)}))
    return tables


def compact_partition(part, sources):
    """Compact a single partition. Called in a forked child process."""
    # import time as _time
    # t0 = _time.monotonic()

    part_dir = PIXEL_DB_PATH / f"part={part}"
    part_dir.mkdir(exist_ok=True, parents=True)
    pq_path = part_dir / "compacted.parquet"
    staging_path = part_dir / "compacted_new.parquet"
    split_marker = part_dir / ".split"

    # skip if already processed in this round (resume after crash, or re-run)
    if staging_path.exists() or split_marker.exists():
        logger.info(f"part {part}: skipping (already processed in this round)")
        return

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

    # t_read = _time.monotonic() - t0

    table = pa.concat_tables(tables)
    # total_rows = len(table)
    # num_sources = len(sources) + (1 if pq_path.exists() else 0)
    del tables

    # sort
    sort_keys = [("hphigh", "ascending")]
    table = table.sort_by(sort_keys)
    sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)

    # t_sort = _time.monotonic() - t0 - t_read

    # check if it's too big
    level, index = part_to_level_index(part)
    if len(table) > MAX_ROWS_PER_PART and level < PART_MAX_LEVEL:
        # split into 4 subpartitions at the next level
        _level = level + 1
        with defer_interrupt():
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
                    use_dictionary=False,
                    sorting_columns=sorting_cols,
                )

            # delete parent's compacted.parquet now that all children are written;
            # touch the .split marker LAST so its presence implies the children
            # are complete (used as the resume marker on re-run).
            pq_path.unlink(missing_ok=True)
            split_marker.touch()

    else:
        with defer_interrupt():
            pq.write_table(
                table,
                staging_path,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
                sorting_columns=sorting_cols,
            )
            # free the old copy immediately to keep peak storage low; the new
            # staging file is the resume marker for re-runs.
            pq_path.unlink(missing_ok=True)

    # t_total = _time.monotonic() - t0
    # t_write = t_total - t_read - t_sort
    # logger.info(
    #     f"part {part}: {total_rows} rows ({num_sources} sources), "
    #     f"read {t_read:.1f}s, sort {t_sort:.1f}s, write {t_write:.1f}s, total {t_total:.1f}s"
    # )


def main():
    # scan binary chunk files for partition boundaries
    partition_index = scan_chunk_files()

    keys = sorted(partition_index.keys())
    if num_tasks > 1:
        keys = keys[task_id::num_tasks]

    iterator = tqdm(keys, desc="task 0", unit="part") if task_id == 0 else keys

    for part in iterator:
        sources = partition_index[part]
        if len(sources) == 0:
            continue

        child_pid = os.fork()
        if child_pid == 0:
            # child process: compact and exit
            try:
                compact_partition(part, sources)
                os._exit(0)
            except Exception:
                logger.exception(f"failed to compact partition {part}")
                for handler in logger.handlers:
                    handler.flush()
                os._exit(1)

        # parent process: wait for child and check result
        _, status = os.waitpid(child_pid, 0)
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            raise RuntimeError(
                f"compact of partition {part} killed by signal {sig}"
            )
        exit_code = os.WEXITSTATUS(status)
        if exit_code != 0:
            raise RuntimeError(f"compact of partition {part} failed (exit {exit_code})")


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()

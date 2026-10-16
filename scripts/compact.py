import functools
import json
import logging
import numpy as np
import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import resource
import socket
import subprocess
import time
import uuid

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.constants import (
    HP_HIGH_LEVEL,
    MAX_ROWS_PER_PART,
    PART_MAX_LEVEL,
    ROW_GROUP_SIZE,
)
from talltable.partition import part_to_level_index, level_index_to_part
from talltable.paths import COMPACTION_ROUND_PATH, PIXEL_DB_PATH, SCRATCH_DIR
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

# The database lives on a ceph-fuse mount, where every read or write call is a round trip through
# the FUSE daemon (profiled 2026-10-16, see scripts/profile_compact.py):
# - a partition's chunk slices are many tiny reads (~10-30 KB each, 7 per source, up to ~300 sources
#   in the deep fields); issued one at a time they cost ~0.2-0.35 s per source, issued from a thread
#   pool ~16 ms per source.
# - pq.write_table(path) writes in small pieces: ~4x slower than local disk, at ~2.7x the cpu.
#   Writing through a large buffer recovers nearly all of it.
# - reading the existing parquet (~500 row groups x 7 column chunks, many of them tens of KB) is
#   ~2.6x faster with pre_buffer=True, which coalesces those reads; under a 48-task load it matched
#   first copying the file to local disk or reading it through once.
CHUNK_READ_THREADS = 16
WRITE_BUFFER_SIZE = 64 * 1024**2

# a partition's output while it is being written; renamed over compacted.parquet when complete
# (post_compact.py removes any left behind by a killed run)
WRITING_NAME = "compacted_writing.parquet"

# footer key recording the compaction round that last rewrote a partition: on a re-run of the same
# round (resume after a failure), partitions that already have it are skipped
ROUND_KEY = "compaction_round"


@functools.cache
def compaction_provenance():
    """footer metadata recorded by every compaction: when it ran, and with which code.

    these are deliberately different keys from the migration's (`migration`, `release`,
    `migrated_at`, `talltable_git`, `calibration`), which must survive compaction untouched.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"  # never fail a compaction over provenance
    return {
        "last_compacted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_compacted_git": commit,
    }


def footer_metadata(schema):
    """the user-defined footer metadata of a parquet file, given its schema, as a str -> str dict
    (without the ARROW:schema entry that pyarrow manages itself)"""
    return {
        k.decode(): v.decode()
        for k, v in (schema.metadata or {}).items()
        if k != b"ARROW:schema"
    }


def compaction_round():
    """The id of the compaction round for the chunk files in SCRATCH_DIR, created by whichever task
    gets there first and read back by the rest.

    A round's partitions are rewritten in place, each marked with the round id in its footer; a re-run
    skips the marked ones, so no chunk is merged twice. That only holds while the round's chunk files
    are exactly the ones it started with, so the round file records them, and a different set (chunk
    files added or removed mid-round, e.g. a new ingest before post_compact.py finished the last round)
    is refused here instead of risking duplicated or skipped rows.
    """
    chunks = sorted(p.name for p in SCRATCH_DIR.glob("chunk_*.bin"))
    tmp = COMPACTION_ROUND_PATH.with_name(f".{COMPACTION_ROUND_PATH.name}.{socket.gethostname()}.{os.getpid()}")
    tmp.write_text(json.dumps({"id": uuid.uuid4().hex, "chunks": chunks}))
    try:
        os.link(tmp, COMPACTION_ROUND_PATH)  # atomic: one task's file wins, complete
    except FileExistsError:
        pass
    finally:
        tmp.unlink()

    round_info = json.loads(COMPACTION_ROUND_PATH.read_text())
    if round_info["chunks"] != chunks:
        added = sorted(set(chunks) - set(round_info["chunks"]))
        missing = sorted(set(round_info["chunks"]) - set(chunks))
        raise RuntimeError(
            f"the chunk files in {SCRATCH_DIR} differ from those of compaction round {round_info['id']} "
            f"({COMPACTION_ROUND_PATH}): {len(added)} added (e.g. {added[:3]}), {len(missing)} missing "
            f"(e.g. {missing[:3]}). Finish or inspect that round by hand before compacting."
        )
    return round_info["id"]


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
        logger.info(
            f"scanned {len(bin_files)} chunk files, found {len(partition_index)} partitions"
        )
    return partition_index


def _pread_exact(fd, nbytes, offset, fpath):
    """os.pread, looping over short reads"""
    parts = []
    while nbytes > 0:
        buf = os.pread(fd, nbytes, offset)
        if not buf:
            raise OSError(f"unexpected end of file in {fpath} at offset {offset}")
        parts.append(buf)
        nbytes -= len(buf)
        offset += len(buf)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def read_source(source):
    """Read one partition slice [start, end) of a binary chunk file, as a table."""
    fpath, start, end = source
    fd = os.open(fpath, os.O_RDONLY)
    try:
        num_part = int(np.frombuffer(_pread_exact(fd, 4, 0, fpath), dtype=np.uint32)[0])
        header_size = 4 + num_part * 20 + 8
        num_rows = int(np.frombuffer(_pread_exact(fd, 8, header_size - 8, fpath), dtype=np.uint64)[0])

        data = {}
        col_offset = header_size
        for name, dtype in CHUNK_COLUMNS:
            itemsize = np.dtype(dtype).itemsize
            buf = _pread_exact(fd, (end - start) * itemsize, col_offset + start * itemsize, fpath)
            data[name] = np.frombuffer(buf, dtype=dtype)
            col_offset += num_rows * itemsize
    finally:
        os.close(fd)
    return pa.table({k: data[k] for k in sorted(data)})


def read_partition_data(sources):
    """Read a partition's data from binary chunk files using contiguous slices, many at once."""
    if len(sources) <= 1:
        return [read_source(s) for s in sources]
    with ThreadPoolExecutor(min(CHUNK_READ_THREADS, len(sources))) as ex:
        return list(ex.map(read_source, sources))


def merge_into(new_tables, existing, outputs, metadata):
    """Merge new rows into an existing partition file, streaming, and write the result.

    `existing` (a ParquetFile, or None for a new partition) is already sorted by hphigh, so only the
    new rows (~1% of a partition per week) are sorted; the existing row groups are then read in
    order, each with the new rows that fall in its hphigh range spliced in, and written out in
    ROW_GROUP_SIZE-row groups as they fill. Memory is about a row group plus the new rows, instead of
    ~2.3x the whole partition for a concat + sort.

    `outputs` is a list of (path, hi): output k gets the rows with hphigh below its `hi` (and at or
    above the previous output's); the last output takes everything that is left. One output for a
    normal compaction, four for a split.

    Rows come out in the same order as a stable sort of concat(new tables..., existing): on equal
    hphigh, new rows first (in source order), then existing ones. The files are byte-identical to
    what pq.write_table of that sorted table writes (verified 2026-10-16, see profile_compact.py).

    Returns (wall seconds spent reading / merging / writing, rows written to each output).
    """
    sort_keys = [("hphigh", "ascending")]
    spent = {"read": 0.0, "merge": 0.0, "write": 0.0}
    clock = time.monotonic()

    def lap(step):
        nonlocal clock
        now = time.monotonic()
        spent[step] += now - clock
        clock = now

    schema = (existing.schema_arrow if existing is not None else new_tables[0].schema).remove_metadata()
    if new_tables:
        new = pa.concat_tables([t.select(schema.names) for t in new_tables]).combine_chunks()
        new = new.take(pc.sort_indices(new, sort_keys=sort_keys))  # stable
    else:
        new = schema.empty_table()
    new_hphigh = new["hphigh"].to_numpy()
    # every image is ingested once, so the existing rows must not contain any image being merged
    # in; if they do, these rows were merged before, and merging them again would duplicate them
    new_images = pa.array(np.unique(new["imageid"].to_numpy()))
    lap("merge")

    sorting_cols = pq.SortingColumn.from_ordering(schema, sort_keys)
    sinks, writers = [], []
    for path, _ in outputs:
        sinks.append(pa.BufferedOutputStream(pa.OSFile(str(path), "wb"), buffer_size=WRITE_BUFFER_SIZE))
        writers.append(
            pq.ParquetWriter(
                sinks[-1],
                schema.with_metadata(metadata),
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
                sorting_columns=sorting_cols,
            )
        )
    pending = [[] for _ in outputs]  # per output: rows not yet written, fewer than ROW_GROUP_SIZE
    pending_rows = [0] * len(outputs)
    rows_written = [0] * len(outputs)

    def emit(k, table, final=False):
        """queue rows for output k, writing every full ROW_GROUP_SIZE-row group (and, if final, the rest)"""
        if len(table):
            pending[k].append(table)
            pending_rows[k] += len(table)
        if pending_rows[k] >= ROW_GROUP_SIZE or (final and pending_rows[k]):
            buf = pa.concat_tables(pending[k]).combine_chunks()
            n_groups = -(-len(buf) // ROW_GROUP_SIZE) if final else len(buf) // ROW_GROUP_SIZE
            for i in range(n_groups):
                group = buf.slice(i * ROW_GROUP_SIZE, ROW_GROUP_SIZE)
                writers[k].write_table(group, row_group_size=ROW_GROUP_SIZE)
                rows_written[k] += len(group)
            rest = buf.slice(n_groups * ROW_GROUP_SIZE)
            pending[k], pending_rows[k] = ([rest] if len(rest) else []), len(rest)

    def route(window):
        """send a sorted window of rows to the outputs whose hphigh ranges it covers"""
        hphigh = window["hphigh"].to_numpy()
        lo = 0
        for k, (_, hi) in enumerate(outputs):
            last = k == len(outputs) - 1
            j = len(hphigh) if last else int(np.searchsorted(hphigh, hi, side="left"))
            if j > lo:
                emit(k, window.slice(lo, j - lo))
            lo = j

    try:
        cursor = 0  # new rows before this index have been merged
        n_groups = existing.num_row_groups if existing is not None else 0
        for i in range(n_groups):
            group = existing.read_row_group(i, use_threads=False).replace_schema_metadata(None)
            lap("read")
            if len(new_images) and pc.any(pc.is_in(group["imageid"], value_set=new_images)).as_py():
                dup = pc.filter(group["imageid"], pc.is_in(group["imageid"], value_set=new_images))[0]
                raise RuntimeError(
                    f"the existing partition file already has rows of image {dup}, which is among the "
                    f"images being merged in: refusing to merge it twice"
                )
            if i == n_groups - 1:
                j = len(new)  # the rest, including any above the file's last hphigh
            elif len(group):
                j = int(np.searchsorted(new_hphigh, group["hphigh"][-1].as_py(), side="right"))
            else:
                j = cursor
            if j > cursor:
                window = pa.concat_tables([new.slice(cursor, j - cursor), group])
                window = window.take(pc.sort_indices(window, sort_keys=sort_keys))  # stable
                cursor = j
            else:
                window = group
            lap("merge")
            route(window)
            lap("write")
        if n_groups == 0:
            route(new)
        for k in range(len(outputs)):
            emit(k, new.schema.empty_table(), final=True)
    finally:
        for writer, sink in zip(writers, sinks):
            writer.close()
            sink.close()
    lap("write")
    return spent, rows_written


def compact_partition(part, sources, round_id):
    """Compact a single partition: merge its new rows into its compacted.parquet, replacing the
    file in place (or, past MAX_ROWS_PER_PART, into its four children). Called in a forked child.

    The partition always has a complete compacted.parquet for queries to read: the new file is
    written under WRITING_NAME and renamed over the old one only once complete. Its footer records
    `round_id`, which is how a re-run of the round knows to skip it.

    A split writes the children's compacted.parquet in their new directories and touches .split in
    the parent's; the parent's own file stays (and is served, as the children aren't in parts.txt
    yet) until post_compact.py removes it and rewrites parts.txt.
    """
    t0 = time.monotonic()
    c0 = time.process_time()

    part_dir = PIXEL_DB_PATH / f"part={part}"
    part_dir.mkdir(exist_ok=True, parents=True)
    pq_path = part_dir / "compacted.parquet"
    split_marker = part_dir / ".split"

    # the existing compacted parquet, if present; its footer metadata (e.g. the QR2 -> QR3
    # migration marker) is carried over
    existing = None
    footer = {}
    if pq_path.exists():
        try:
            existing = pq.ParquetFile(pq_path, pre_buffer=True)
            footer = footer_metadata(existing.schema_arrow)
        except pa.lib.ArrowInvalid as e:
            msg = f"failed to open Parquet file {pq_path} with error message:\n{e}"
            raise RuntimeError(msg)

    # skip if already done in this round (resume after a failure, or re-run)
    if split_marker.exists() or footer.get(ROUND_KEY) == round_id:
        logger.info(f"part {part}: skipping (already compacted in this round)")
        return

    # flat chunk files (contiguous slices)
    new_tables = read_partition_data(sources) if sources else []
    t_chunks = time.monotonic() - t0

    total_rows = sum(len(t) for t in new_tables) + (existing.metadata.num_rows if existing else 0)
    num_sources = len(sources) + (1 if existing else 0)

    # if it's too big, split into 4 subpartitions at the next level
    level, index = part_to_level_index(part)
    split = total_rows > MAX_ROWS_PER_PART and level < PART_MAX_LEVEL
    if split:
        targets = []
        for k in range(4):
            _index = (index << 2) + k
            _part_dir = PIXEL_DB_PATH / f"part={level_index_to_part(level + 1, _index)}"
            _part_dir.mkdir(exist_ok=True, parents=True)
            _hi = (_index + 1) << 2 * (HP_HIGH_LEVEL - level - 1)
            targets.append((_part_dir / "compacted.parquet", _hi))
    else:
        targets = [(pq_path, None)]

    # write under a temporary name, then rename over compacted.parquet only once complete
    writing = [(path.with_name(WRITING_NAME), hi) for path, hi in targets]
    metadata = {**footer, **compaction_provenance(), ROUND_KEY: round_id}
    spent, rows_written = merge_into(new_tables, existing, writing, metadata)
    if sum(rows_written) != total_rows:
        raise RuntimeError(f"part {part}: wrote {sum(rows_written)} rows, expected {total_rows}")

    with defer_interrupt():
        for (tmp, _), (path, _) in zip(writing, targets):
            tmp.replace(path)
        if split:
            # touched LAST, so its presence implies the children are complete
            split_marker.touch()

    # cpu well below total wall time means the partition spent its time waiting (on I/O)
    t_total = time.monotonic() - t0
    cpu = time.process_time() - c0
    maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    logger.info(
        f"part {part}: {total_rows} rows ({num_sources} sources), chunks {t_chunks:.1f}s, "
        f"read {spent['read']:.1f}s, merge {spent['merge']:.1f}s, write {spent['write']:.1f}s, "
        f"total {t_total:.1f}s, cpu {cpu:.1f}s, maxrss {maxrss_gb:.1f}GB" + (" (split)" if split else "")
    )


def main():
    # compute the footer provenance once, here, so the forked children below reuse it
    compaction_provenance()

    # scan binary chunk files for partition boundaries
    partition_index = scan_chunk_files()
    round_id = compaction_round()
    if task_id == 0:
        logger.info(f"compaction round {round_id}")

    keys = sorted(partition_index.keys())
    if num_tasks > 1:
        keys = keys[task_id::num_tasks]

    iterator = tqdm(keys, desc="task 0", unit="part") if task_id == 0 else keys
    t_start = time.monotonic()

    for part in iterator:
        sources = partition_index[part]
        if len(sources) == 0:
            continue

        child_pid = os.fork()
        if child_pid == 0:
            # child process: compact and exit
            try:
                compact_partition(part, sources, round_id)
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
            raise RuntimeError(f"compact of partition {part} killed by signal {sig}")
        exit_code = os.WEXITSTATUS(status)
        if exit_code != 0:
            raise RuntimeError(f"compact of partition {part} failed (exit {exit_code})")

    # per-task wall time: the spread across tasks is the load imbalance of the striding
    logger.info(
        f"task done on {socket.gethostname()}: {len(keys)} partitions "
        f"in {(time.monotonic() - t_start) / 3600:.2f} h"
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()

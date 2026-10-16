"""
profile_compact.py
author: Connor Hainje

Profile compact.py on real partitions without touching the database: it reads the real
compacted.parquet files and writes everything else under a scratch work directory.

Run the stages in order, on a compute node (from this directory, with `uv run python`):

  select    read the WCS footprint of every image in a week, find the partitions the week
            touches (touched.csv), and pick a size-stratified sample of them for `run`. Also
            prints how much existing data the week will rewrite.
  chunks    write binary chunk files for every image touching the picked partitions, grouped
            exactly as the real ingest groups them (same sorted file list, same task striding,
            same chunk size), so the picked partitions' chunk slices are identical to the ones
            the real compaction will see.
  run       compact the picked partitions with per-step wall and cpu times, peak memory, and
            sizes. One worker by default, pinned to one core like a production srun task.
  contend   compact partitions (existing data only, a fresh random sample of touched.csv per
            variant) at several concurrency levels, to tell filesystem contention apart from
            per-partition CPU cost.
  load      compact 48 adjacent partitions with real chunk data at once (one pinned task each,
            like a production node), from sky regions nothing has read yet; for comparing
            --chunks / --write / --read variants under production load.
  summary   re-print the results of earlier `run` / `contend` stages.

For a flame graph of one partition (python and pyarrow C++ frames):
  uvx py-spy record --native --subprocesses -o compact.svg -- \
      python profile_compact.py run --parts <part>

Input page caches are dropped (posix_fadvise) before each partition, best effort; the Ceph
servers may still have recently read files cached, so repeated runs on the same partitions
can come out optimistic.
"""

import json
import os
import sys
import time
import warnings

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

DEFAULT_WORKDIR = Path("/mnt/sdceph/users/chainje/compact-profile")
RELEASE_DATA_DIR_VARS = {"qr2": "TALLTABLE_DATA_DIR", "qr3": "TALLTABLE_DATA_DIR_QR3"}


def parse():
    ap = ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    sub = ap.add_subparsers(dest="stage", required=True)

    sp = sub.add_parser("select")
    sp.add_argument("--week", required=True, help="e.g. 2026W31")
    sp.add_argument("--release", choices=RELEASE_DATA_DIR_VARS, default="qr3")
    sp.add_argument("--filelist", type=Path, help="the week's file list (default: find it like ingest_week.sbatch)")
    sp.add_argument("--ntasks", type=int, default=96, help="ingest tasks in production (2 nodes x 48)")
    sp.add_argument("--chunk-size", type=int, default=24, help="ingest.py -C in production")
    sp.add_argument(
        "--quantiles", default="0.1,0.3,0.5,0.7,0.9,0.97,0.995,1.0",
        help="size quantiles of the partitions to pick for `run`",
    )
    sp.add_argument("-n", "--workers", type=int, default=os.cpu_count())

    sp = sub.add_parser("chunks")
    sp.add_argument("-n", "--workers", type=int, default=os.cpu_count())
    sp.add_argument("--force", action="store_true", help="delete existing test chunk files first")

    sp = sub.add_parser("run")
    sp.add_argument("--parts", type=int, nargs="+", help="subset of the picked partitions")
    sp.add_argument("-c", "--concurrency", type=int, default=1)
    sp.add_argument("--no-pin", action="store_true", help="let workers use all cores")
    sp.add_argument("--keep", action="store_true", help="keep the written parquet files")
    add_io_args(sp)

    sp = sub.add_parser("contend")
    sp.add_argument("--levels", default="1,12,48", help="concurrency levels")
    sp.add_argument("--parts-per-task", type=int, default=1)
    sp.add_argument("--no-pin", action="store_true")
    sp.add_argument("--seed", type=int, help="partition sample (default: different for every --read/--write variant)")
    add_io_args(sp)

    sp = sub.add_parser("load")
    sp.add_argument("-c", "--concurrency", type=int, default=48)
    sp.add_argument("--parts-per-task", type=int, default=1)
    sp.add_argument("--min-sources", type=int, default=60, help="only partitions with this many chunk sources")
    sp.add_argument("--no-pin", action="store_true")
    add_io_args(sp)

    sub.add_parser("summary")
    return ap.parse_args()


def add_io_args(sp):
    sp.add_argument(
        "--read", choices=["direct", "prebuffer", "warm", "stage"], default="direct",
        help="how to read the existing parquet: as compact.py does (direct), with pyarrow's "
        "coalesced reads (prebuffer), after one sequential pass over the file to pull it into the "
        "page cache (warm), or copied to local disk first (stage)",
    )
    sp.add_argument(
        "--write", choices=["direct", "buffered", "stage"], default="direct",
        help="write the output with pq.write_table(path) (direct, compact.py before 2026-10-16), "
        "through a WRITE_BUFFER_SIZE buffer (buffered, compact.py), or to local disk then copy (stage)",
    )
    sp.add_argument(
        "--chunks", choices=["sequential", "threaded"], default="sequential",
        help="read chunk slices one at a time (sequential, compact.py before 2026-10-16) or with "
        "compact.read_partition_data's thread pool (threaded)",
    )
    sp.add_argument(
        "--merge", choices=["full", "stream"], default="full",
        help="load everything, sort, write (full, as compact.py) or merge the sorted new rows into a "
        "row-group-by-row-group stream of the existing file (stream; --read direct|prebuffer only)",
    )
    sp.add_argument("--local-dir", type=Path, default=Path("/tmp/compact-profile"))


def io_label(args):
    return f"-r{args.read}-w{args.write}-c{args.chunks}" + ("-mstream" if args.merge == "stream" else "")


def read_partition_data_sequential(sources):
    """compact.read_partition_data before 2026-10-16: one slice at a time, through np.fromfile"""
    import numpy as np
    import pyarrow as pa
    from talltable_pipeline.batch import CHUNK_COLUMNS

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


# ---------------------------------------------------------------------------------------------
# select


_parts = None  # active partitions, set before forking the footprint workers


def footprint_parts(path):
    """the active partitions an image's footprint touches (generously: inclusive at level 10)"""
    import healpy as hp
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from talltable.constants import PART_MAX_LEVEL, PART_MIN_LEVEL
    from talltable.partition import level_index_to_part

    try:
        hdr = fits.getheader(path, "IMAGE")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(hdr)
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
        xs = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        ys = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
        ra, dec = wcs.all_pix2world(xs, ys, 0)
        pix = hp.query_polygon(
            2**PART_MAX_LEVEL, hp.ang2vec(ra, dec, lonlat=True), inclusive=True, nest=True
        )
    except Exception as err:
        print(f"warning: no footprint for {path}: {err}", file=sys.stderr)
        return path, []

    # walk each level-10 pixel up to its active partition, like BatchWriter.process_image
    found = set()
    for p in level_index_to_part(PART_MAX_LEVEL, pix):
        p = int(p)
        for _ in range(PART_MAX_LEVEL - PART_MIN_LEVEL):
            if p in _parts:
                break
            p >>= 2
        found.add(p)
    return path, sorted(found)


def stage_select(args):
    global _parts
    import numpy as np
    from tqdm import tqdm
    from talltable.paths import PART_DB_PATH, PIXEL_DB_PATH, require_env

    year, week = args.week.upper().split("W")
    week = f"{int(week):02d}"

    # the same file list, in the same order, as ingest_week.sbatch + ingest.py
    if args.filelist:
        files = [line.strip() for line in open(args.filelist) if line.strip()]
    else:
        level2 = require_env(RELEASE_DATA_DIR_VARS[args.release]) / "level2"
        files = [str(p) for p in level2.glob(f"*/*/*/*{year}W{week}_*.fits")]
    files = sorted(files)
    print(f"{year}W{week}: {len(files)} files")

    with open(PART_DB_PATH) as f:
        _parts = set(int(line) for line in f if line.strip())

    part_images = {}  # partition -> indices (into `files`) of the images touching it
    index = {p: i for i, p in enumerate(files)}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("fork")) as ex:
        results = ex.map(footprint_parts, files, chunksize=16)
        for path, parts in tqdm(results, total=len(files), desc="footprints"):
            for p in parts:
                part_images.setdefault(p, []).append(index[path])

    def size(p):
        try:
            return (PIXEL_DB_PATH / f"part={p}" / "compacted.parquet").stat().st_size
        except FileNotFoundError:
            return 0

    touched = sorted(part_images)
    with ThreadPoolExecutor(32) as ex:
        sizes = dict(zip(touched, ex.map(size, touched)))

    total = sum(sizes.values())
    print(
        f"the week touches {len(touched)} of {len(_parts)} partitions, "
        f"holding {total / 1e12:.1f} TB of existing compacted parquet "
        f"(median {np.median(list(sizes.values())) / 1e9:.2f} GB, max {max(sizes.values()) / 1e9:.2f} GB)"
    )

    # size-stratified pick for `run`
    existing = sorted((p for p in touched if sizes[p] > 0), key=lambda p: sizes[p])
    qs = [float(q) for q in args.quantiles.split(",")]
    picked = sorted(set(existing[round(q * (len(existing) - 1))] for q in qs), key=lambda p: sizes[p])

    needed = sorted(set(i for p in picked for i in part_images[p]))
    selection = {
        "week": f"{year}W{week}",
        "release": args.release,
        "ntasks": args.ntasks,
        "chunk_size": args.chunk_size,
        "num_files": len(files),
        "touched_bytes": total,
        "picked": [{"part": p, "bytes": sizes[p], "images": part_images[p]} for p in picked],
        "images": {str(i): files[i] for i in needed},
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    with open(args.workdir / "selection.json", "w") as f:
        json.dump(selection, f)
    with open(args.workdir / "touched.csv", "w") as f:
        f.write("part,bytes,num_images\n")
        for p in touched:
            f.write(f"{p},{sizes[p]},{len(part_images[p])}\n")

    print("\npicked for `run`:")
    for p in picked:
        print(f"  part {p}: {sizes[p] / 1e9:6.2f} GB, {len(part_images[p])} images")
    print(f"{len(needed)} images to ingest in `chunks`")
    print(f"wrote {args.workdir / 'selection.json'} and {args.workdir / 'touched.csv'}")


# ---------------------------------------------------------------------------------------------
# chunks


def write_chunk(key, paths):
    """one production chunk file, restricted to the given images (in production order)"""
    from talltable_pipeline.batch import BatchWriter

    task, chunk = key
    batch = BatchWriter(chunk_size=len(paths) + 1, auto_write=False, task_id=f"{task}c{chunk}")
    for path in paths:
        batch.process_image(path)
    if batch.count() > 0:
        batch._write_pixels()  # not batch.write(): that would also write image parts
    return len(paths)


def stage_chunks(args):
    from tqdm import tqdm
    from talltable.paths import SCRATCH_DIR

    sel = load_selection(args)
    existing = list(SCRATCH_DIR.glob("chunk_*.bin"))
    if existing and not args.force:
        sys.exit(f"{len(existing)} chunk files already in {SCRATCH_DIR}; use --force to replace them")
    for c in existing:
        c.unlink()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # production: ingest.py strides the sorted list over tasks, then each task writes a chunk
    # every `chunk_size` images
    ntasks, chunk_size = sel["ntasks"], sel["chunk_size"]
    groups = {}
    for i in sorted(int(i) for i in sel["images"]):
        key = (i % ntasks, (i // ntasks) // chunk_size)
        groups.setdefault(key, []).append(sel["images"][str(i)])

    print(f"writing {len(groups)} chunk files ({len(sel['images'])} images) to {SCRATCH_DIR}")
    with ProcessPoolExecutor(args.workers, mp_context=get_context("fork")) as ex:
        futures = [ex.submit(write_chunk, k, v) for k, v in groups.items()]
        with tqdm(total=len(sel["images"]), unit="image") as pbar:
            for fut in futures:
                pbar.update(fut.result())


# ---------------------------------------------------------------------------------------------
# run / contend


def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def timed_compact(part, sources, out_dir, keep, io):
    """compact.compact_partition step by step, timed, with the output going to `out_dir`.
    Mirrors its calls exactly (sort_by is sort_indices + take, timed separately here) when
    io = {"read": "direct", "write": "direct"}."""
    import shutil
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from talltable.constants import HP_HIGH_LEVEL, MAX_ROWS_PER_PART, PART_MAX_LEVEL, ROW_GROUP_SIZE
    from talltable.partition import level_index_to_part, part_to_level_index
    from talltable.paths import PIXEL_DB_PATH
    from compact import WRITE_BUFFER_SIZE, footer_metadata, read_partition_data

    def write_parquet(table, path, **kwargs):
        with pa.BufferedOutputStream(pa.OSFile(str(path), "wb"), buffer_size=WRITE_BUFFER_SIZE) as out:
            pq.write_table(table, out, **kwargs)

    rec = {"part": part, "steps": {}, "pid": os.getpid(), "cpus": len(os.sched_getaffinity(0)), **io}
    local = Path(io["local_dir"]) / f"part={part}"
    t_wall, t_cpu = time.monotonic(), time.process_time()

    def lap(name):
        nonlocal t_wall, t_cpu
        now_wall, now_cpu = time.monotonic(), time.process_time()
        rec["steps"][name] = {"wall": now_wall - t_wall, "cpu": now_cpu - t_cpu}
        t_wall, t_cpu = now_wall, now_cpu

    reader = read_partition_data if io.get("chunks") == "threaded" else read_partition_data_sequential
    tables = reader(sources) if sources else []
    rec["rows_new"] = sum(len(t) for t in tables)
    rec["num_sources"] = len(sources)
    lap("read_chunks")

    pq_path = PIXEL_DB_PATH / f"part={part}" / "compacted.parquet"
    rec["bytes_existing"] = pq_path.stat().st_size
    if io.get("merge") == "stream":
        outputs = stream_merge(part, tables, pq_path, out_dir, io, rec, lap)
        return finish(rec, outputs, keep)
    if io["read"] == "stage":
        local.mkdir(parents=True, exist_ok=True)
        staged = local / "existing.parquet"
        shutil.copyfile(pq_path, staged)
        lap("stage_in")
        existing = pq.ParquetFile(staged)
    elif io["read"] == "warm":
        buf = bytearray(16 * 1024**2)
        with open(pq_path, "rb", buffering=0) as f:
            while f.readinto(buf):
                pass
        del buf
        lap("warm")
        existing = pq.ParquetFile(pq_path)
    else:
        existing = pq.ParquetFile(pq_path, pre_buffer=io["read"] == "prebuffer")
    footer = footer_metadata(existing.schema_arrow)
    rec["row_groups_existing"] = existing.num_row_groups
    tables.append(existing.read())
    rec["rows_existing"] = len(tables[-1])
    if io["read"] == "stage":
        existing.close()
        staged.unlink()
    lap("read_existing")

    table = pa.concat_tables(tables)
    del tables
    lap("concat")

    sort_keys = [("hphigh", "ascending")]
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    lap("sort_indices")
    table = table.take(indices)
    del indices
    table = table.replace_schema_metadata(footer)
    sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)
    lap("take")

    write_kw = dict(
        compression="zstd", compression_level=3, use_dictionary=False,
        sorting_columns=sorting_cols, row_group_size=ROW_GROUP_SIZE,
    )
    # (destination, where to write it first) per output file
    outputs = []
    level, index = part_to_level_index(part)
    rec["split"] = len(table) > MAX_ROWS_PER_PART and level < PART_MAX_LEVEL
    if rec["split"]:
        pieces = []
        for k in range(4):
            _index = (index << 2) + k
            _lo = _index << 2 * (HP_HIGH_LEVEL - level - 1)
            _hi = (_index + 1) << 2 * (HP_HIGH_LEVEL - level - 1)
            _table = table.filter((pc.field("hphigh") >= _lo) & (pc.field("hphigh") < _hi))
            pieces.append((level_index_to_part(level + 1, _index), _table))
    else:
        pieces = [(part, table)]
    for p, t in pieces:
        out = out_dir / f"part={p}" / "compacted_new.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if io["write"] == "stage":
            local.mkdir(parents=True, exist_ok=True)
            first = local / f"part={p}.parquet"
        else:
            first = out
        if io["write"] == "buffered":
            write_parquet(t, first, **write_kw)
        else:
            pq.write_table(t, first, **write_kw)
        # the file must read back complete (cheap: footer only)
        assert pq.ParquetFile(first).metadata.num_rows == len(t), f"{first}: wrong row count"
        outputs.append((out, first))
    lap("write")

    if io["write"] == "stage":
        for out, first in outputs:
            shutil.copyfile(first, out)
            first.unlink()
        lap("stage_out")
    if local.exists():
        local.rmdir()
    return finish(rec, [out for out, _ in outputs], keep)


def finish(rec, outputs, keep):
    import resource

    rec["bytes_out"] = sum(o.stat().st_size for o in outputs)
    rec["maxrss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    rec["wall"] = sum(s["wall"] for s in rec["steps"].values())
    rec["cpu"] = sum(s["cpu"] for s in rec["steps"].values())
    if not keep:
        for o in outputs:
            o.unlink()
            o.parent.rmdir()
    return rec


def stream_merge(part, new_tables, pq_path, out_dir, io, rec, lap):
    """compaction through compact.merge_into (the streaming merge compact.py uses), output to
    `out_dir`; with the same footer as the full-merge path here, so their outputs can be compared
    byte for byte"""
    import pyarrow.parquet as pq
    from talltable import constants
    from talltable.partition import level_index_to_part, part_to_level_index
    from compact import footer_metadata, merge_into

    existing = pq.ParquetFile(pq_path, pre_buffer=io["read"] == "prebuffer")
    rec["rows_new"] = sum(len(t) for t in new_tables)
    rec["rows_existing"] = existing.metadata.num_rows
    rec["row_groups_existing"] = existing.num_row_groups
    total = rec["rows_existing"] + rec["rows_new"]

    level, index = part_to_level_index(part)
    rec["split"] = total > constants.MAX_ROWS_PER_PART and level < constants.PART_MAX_LEVEL
    if rec["split"]:
        shift = 2 * (constants.HP_HIGH_LEVEL - level - 1)
        targets = [
            (out_dir / f"part={level_index_to_part(level + 1, (index << 2) + k)}" / "compacted_new.parquet",
             ((index << 2) + k + 1) << shift)
            for k in range(4)
        ]
    else:
        targets = [(out_dir / f"part={part}" / "compacted_new.parquet", None)]
    for path, _ in targets:
        path.parent.mkdir(parents=True, exist_ok=True)

    spent, rows_written = merge_into(new_tables, existing, targets, footer_metadata(existing.schema_arrow))
    lap("stream")  # read + merge + write, interleaved; the wall-time split is in rec["stream_split"]
    rec["stream_split"] = spent
    assert sum(rows_written) == total, f"part {part}: wrote {sum(rows_written)} rows, expected {total}"
    return [path for path, _ in targets]


def task_loop(k, parts, sources_by_part, cpu, out_dir, result_dir, keep, io):
    """one production-like compact task: pinned to a core, forking a child per partition"""
    from talltable.paths import PIXEL_DB_PATH

    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    for part in parts:
        sources = sources_by_part.get(part, [])
        drop_cache(PIXEL_DB_PATH / f"part={part}" / "compacted.parquet")
        for fpath, _, _ in sources:
            drop_cache(fpath)

        pid = os.fork()
        if pid == 0:
            try:
                rec = timed_compact(part, sources, out_dir, keep, io)
                rec["task"] = k
                with open(result_dir / f"part={part}.json", "w") as f:
                    json.dump(rec, f)
                os._exit(0)
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        if status != 0:
            print(f"task {k}: partition {part} failed (status {status})", file=sys.stderr)


def run_tasks(args, label, assignments, sources_by_part, pin, keep=False):
    """run one production-like task per entry of `assignments` concurrently"""
    import pyarrow as pa

    label += io_label(args)
    io = {"read": args.read, "write": args.write, "chunks": args.chunks, "merge": args.merge, "local_dir": str(args.local_dir)}
    result_dir = args.workdir / "results" / label
    result_dir.mkdir(parents=True, exist_ok=True)
    for old in result_dir.glob("*.json"):
        old.unlink()
    out_dir = args.workdir / "out"

    cores = sorted(os.sched_getaffinity(0))
    if pin and len(assignments) > len(cores):
        sys.exit(f"{len(assignments)} pinned tasks but only {len(cores)} cores")
    print(
        f"\n[{label}] {len(assignments)} task(s), {sum(map(len, assignments))} partition(s), "
        f"{'pinned to one core each' if pin else 'unpinned'} "
        f"(pyarrow cpu_count={pa.cpu_count()}, io_thread_count={pa.io_thread_count()})"
    )

    ctx = get_context("fork")
    t0 = time.monotonic()
    procs = [
        ctx.Process(
            target=task_loop,
            args=(k, parts, sources_by_part, cores[k] if pin else None, out_dir, result_dir, keep, io),
        )
        for k, parts in enumerate(assignments)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.monotonic() - t0
    with open(result_dir / "_meta.json", "w") as f:
        json.dump({"label": label, "tasks": len(assignments), "pin": pin, "elapsed": elapsed}, f)
    summarize(result_dir)


def chunk_sources(sel):
    """the picked partitions' chunk slices, as compact.scan_chunk_files finds them"""
    from talltable.paths import SCRATCH_DIR
    from compact import scan_chunk_files

    index = scan_chunk_files()
    picked = [d["part"] for d in sel["picked"]]
    missing = [p for p in picked if p not in index]
    if not list(SCRATCH_DIR.glob("chunk_*.bin")):
        sys.exit(f"no chunk files in {SCRATCH_DIR}; run the `chunks` stage first")
    if missing:
        print(f"warning: no chunk data for partitions {missing}", file=sys.stderr)
    return {p: index[p] for p in picked if p in index}


def stage_run(args):
    sel = load_selection(args)
    parts = args.parts or [d["part"] for d in sel["picked"]]
    sources = chunk_sources(sel)
    c = args.concurrency
    assignments = [parts[k::c] for k in range(c) if parts[k::c]]
    label = f"run-c{c}-{'nopin' if args.no_pin else 'pin'}"
    run_tasks(args, label, assignments, sources, pin=not args.no_pin, keep=args.keep)


def stage_contend(args):
    import zlib
    import numpy as np

    sel = load_selection(args)
    levels = [int(x) for x in args.levels.split(",")]
    need = sum(levels) * args.parts_per_task

    # a fresh random sample of the week's partitions for every variant (seeded by the variant), so
    # no variant reads files that an earlier one left warm in Ceph's caches
    picked = {d["part"] for d in sel["picked"]}
    with open(args.workdir / "touched.csv") as f:
        next(f)
        candidates = [int(p) for p, b, _ in (line.split(",") for line in f) if int(b) > 0 and int(p) not in picked]
    seed = args.seed if args.seed is not None else zlib.crc32(io_label(args).encode())
    pool = [int(p) for p in np.random.default_rng(seed).choice(candidates, size=need, replace=False)]
    print(f"contend: {need} partitions sampled with seed {seed}")

    # distinct partitions at every level, so nothing is read twice
    for c in levels:
        take, pool = pool[: c * args.parts_per_task], pool[c * args.parts_per_task :]
        assignments = [take[k::c] for k in range(c)]
        label = f"contend-c{c}-{'nopin' if args.no_pin else 'pin'}"
        run_tasks(args, label, assignments, {}, pin=not args.no_pin)


def region(part):
    """the level-6 sky region a partition lies in"""
    n = part.bit_length() - 1
    level = n // 2 - 4
    return (part - (1 << n)) >> (2 * (level - 6))


def stage_load(args):
    """production-like load with real chunk data: `concurrency` pinned tasks at once on adjacent
    partitions (as compact.py's striding hands a node's tasks neighbouring keys), from sky regions
    no earlier measurement has read (ceph-fuse caches and reads ahead, so a region read before would
    come out optimistic)"""
    from talltable.paths import PIXEL_DB_PATH
    from compact import scan_chunk_files

    index = scan_chunk_files()
    used = {d["part"] for d in load_selection(args)["picked"]}
    used |= {int(f.stem.split("=")[1]) for f in (args.workdir / "results").glob("*/part=*.json")}
    used_file = args.workdir / "used_parts.txt"  # partitions read outside this script's results
    if used_file.exists():
        used |= {int(line) for line in open(used_file) if line.strip()}
    used_regions = {region(p) for p in used}

    need = args.concurrency * args.parts_per_task
    take = []
    for p in sorted(index):
        if len(index[p]) < args.min_sources or region(p) in used_regions:
            continue
        if not (PIXEL_DB_PATH / f"part={p}" / "compacted.parquet").exists():
            continue
        take.append(p)
        if len(take) == need:
            break
    if len(take) < need:
        sys.exit(f"only {len(take)} fresh partitions with >= {args.min_sources} sources; need {need}")
    print(f"load: partitions {take[0]}..{take[-1]} ({len({region(p) for p in take})} sky regions)")

    c = args.concurrency
    assignments = [take[k::c] for k in range(c)]
    label = f"load-c{c}-{'nopin' if args.no_pin else 'pin'}"
    run_tasks(args, label, assignments, {p: index[p] for p in take}, pin=not args.no_pin)


# ---------------------------------------------------------------------------------------------
# summary


STEPS = [
    "read_chunks", "warm", "stage_in", "read_existing", "concat", "sort_indices", "take", "write",
    "stage_out", "stream",
]


def summarize(result_dir):
    import numpy as np

    recs = []
    for f in sorted(result_dir.glob("part=*.json")):
        with open(f) as fh:
            recs.append(json.load(fh))
    meta_path = result_dir / "_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    if not recs:
        print(f"[{result_dir.name}] no results")
        return
    recs.sort(key=lambda r: r["bytes_existing"])

    steps = [s for s in STEPS if any(s in r["steps"] for r in recs)]
    print(f"\n[{result_dir.name}]  wall s (cpu/wall) per step")
    header = f"{'part':>9} {'lvl':>3} {'GB':>6} {'Mrows':>6} {'new%':>5} {'src':>4} " + " ".join(f"{s[:13]:>14}" for s in steps)
    header += f" {'total':>14} {'rssGB':>6}"
    print(header)
    for r in recs:
        rows = r["rows_existing"] + r["rows_new"]
        cells = []
        for s in steps + ["total"]:
            st = r["steps"][s] if s != "total" else {"wall": r["wall"], "cpu": r["cpu"]}
            ratio = st["cpu"] / st["wall"] if st["wall"] > 0.05 else float("nan")
            cells.append(f"{st['wall']:8.1f} ({ratio:3.1f})")
        print(
            f"{r['part']:>9} {(r['part'].bit_length() - 1) // 2 - 4:>3} "
            f"{r['bytes_existing'] / 1e9:6.2f} {rows / 1e6:6.1f} "
            f"{100 * r['rows_new'] / max(rows, 1):5.1f} {r['num_sources']:>4} "
            + " ".join(f"{c:>14}" for c in cells)
            + f" {r['maxrss_gb']:6.1f}" + ("  split" if r.get("split") else "")
        )

    gb = sum(r["bytes_existing"] for r in recs) / 1e9
    walls = np.array([r["wall"] for r in recs])
    per_gb = walls / np.array([r["bytes_existing"] / 1e9 for r in recs])
    line = f"  {len(recs)} partitions, {gb:.1f} GB read; median {np.median(per_gb):.1f} s/GB"
    if meta.get("elapsed"):
        line += f"; elapsed {meta['elapsed']:.0f} s, aggregate {gb / meta['elapsed'] * 1e3:.0f} MB/s of existing parquet"
    print(line)


def stage_summary(args):
    for d in sorted((args.workdir / "results").iterdir()):
        if d.is_dir():
            summarize(d)


# ---------------------------------------------------------------------------------------------


def load_selection(args):
    path = args.workdir / "selection.json"
    if not path.exists():
        sys.exit(f"{path} not found; run the `select` stage first")
    with open(path) as f:
        return json.load(f)


def main():
    args = parse()

    # chunk files go under the work directory: set before talltable.paths reads it (a real env
    # var beats .env)
    os.environ["TALLTABLE_SCRATCH_DIR"] = str(args.workdir / "chunks")
    from talltable.paths import DB_DIR, SCRATCH_DIR

    workdir = args.workdir.resolve()
    if workdir == DB_DIR.resolve() or DB_DIR.resolve() in workdir.parents:
        sys.exit(f"the work directory must not be inside the database ({DB_DIR})")
    assert SCRATCH_DIR == args.workdir / "chunks"

    {
        "select": stage_select,
        "chunks": stage_chunks,
        "run": stage_run,
        "contend": stage_contend,
        "load": stage_load,
        "summary": stage_summary,
    }[args.stage](args)


if __name__ == "__main__":
    main()

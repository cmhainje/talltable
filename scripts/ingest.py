"""
ingest.py
author: Connor Hainje

usage:
python ingest.py <filelist> -C <chunk_size>

Supports SLURM parallelism via SLURM_PROCID and SLURM_NTASKS env vars.
Each task processes a strided subset of the file list and writes to its own
intermediate image parquet file. Run compact.py after to merge.
"""

import logging
import os
import queue
import threading

from argparse import ArgumentParser
from os.path import basename
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.batch import BatchWriter, read_image
from talltable.query import get_image_filepaths
from talltable.paths import DATA_DIR, PIXEL_DB_PATH, IMAGE_PARTS_DIR


logger = logging.getLogger(__name__)
task_id = int(os.environ.get("SLURM_PROCID", 0))
num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
job_id = os.environ.get("SLURM_JOB_ID")
out_file = os.environ.get("SLURM_JOB_STDOUT", f"./slurm-{job_id}.out")


def parse():
    ap = ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("-C", "--chunk-size", type=int, nargs="?", default=48)
    ap.add_argument("-F", "--force", action="store_true", help="force (re-ingest)")
    return ap.parse_args()


_DONE = object()  # sentinel to signal end of queue


def reader_worker(filepaths, data_queue):
    """Read-ahead thread: reads FITS files and puts data on the queue."""
    for filepath in filepaths:
        try:
            data = read_image(filepath)
            data_queue.put(data)
        except OSError as err:
            logger.error("error reading %s: %s", filepath, err)
            data_queue.put(None)  # skip marker for failed reads
    data_queue.put(_DONE)


def main(args):
    logger.info("SLURM task %d of %d", task_id, num_tasks)

    if task_id == 0:
        PIXEL_DB_PATH.mkdir(exist_ok=True, parents=True)
        IMAGE_PARTS_DIR.mkdir(exist_ok=True, parents=True)

    if IMAGE_PARTS_DIR.exists() and len(list(IMAGE_PARTS_DIR.glob("*.parquet"))) > 0:
        raise RuntimeError(
            f"there are existing image part files in {IMAGE_PARTS_DIR}! "
            "did you run `compact` after the last `ingest`?"
        )

    with open(args.infile, "r") as f:
        data_files = [str(DATA_DIR / u.strip()) for u in f.readlines()]

    done_before = (
        set(basename(p) for p in get_image_filepaths()) if not args.force else set()
    )
    to_ingest = sorted([p for p in data_files if basename(p) not in done_before])

    if num_tasks > 1:
        to_ingest = to_ingest[task_id::num_tasks]

    logger.info("task %d: %d files to ingest", task_id, len(to_ingest))

    if len(to_ingest) == 0:
        return

    batch = BatchWriter(chunk_size=args.chunk_size, task_id=task_id)

    # read-ahead thread prefetches FITS files while main thread does compute
    data_queue = queue.Queue(maxsize=2)
    reader = threading.Thread(
        target=reader_worker, args=(to_ingest, data_queue), daemon=True
    )
    reader.start()

    with tqdm(total=len(to_ingest)) as pbar:
        while True:
            fits_data = data_queue.get()
            if fits_data is _DONE:
                break
            pbar.update(1)
            if fits_data is None:
                continue  # skip failed reads
            logger.debug("processing %s", basename(fits_data.filepath))
            batch.process_image(fits_data)

    # flush any remaining buffered data
    if batch.count() > 0:
        batch.write()

    reader.join(timeout=5)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename=out_file + f".{task_id}",
        format=f"%(asctime)s [{task_id}] %(levelname)s %(message)s",
        force=True,
    )
    args = parse()
    with logging_redirect_tqdm():
        main(args)

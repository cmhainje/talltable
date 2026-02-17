"""
ingest.py
author: Connor Hainje

usage:
python ingest.py <filelist> -N <workers> -C <chunk_size>

Supports SLURM parallelism via SLURM_PROCID and SLURM_NTASKS env vars.
Each task processes a strided subset of the file list and writes to its own
intermediate image parquet file. Run compact.py after to merge.
"""

import logging
import os

from argparse import ArgumentParser
from glob import glob
from os.path import basename
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.batch import BatchWriter

from talltable.query import get_image_filepaths
from talltable.paths import DATA_DIR, PIXEL_DB_PATH, IMAGE_PARTS_DIR


logger = logging.getLogger(__name__)


def parse():
    ap = ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("-C", "--chunk-size", type=int, nargs="?", default=16)
    ap.add_argument("-F", "--force", action="store_true", help="force (re-ingest)")
    return ap.parse_args()


def main(args):
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    logger.info("SLURM task %d of %d", task_id, num_tasks)

    if task_id == 0:
        PIXEL_DB_PATH.mkdir(exist_ok=True, parents=True)
        IMAGE_PARTS_DIR.mkdir(exist_ok=True, parents=True)

    if IMAGE_PARTS_DIR.exists() and len(glob(str(IMAGE_PARTS_DIR / "*.parquet"))) > 0:
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

    for index in tqdm(range(len(to_ingest)), desc=f"task {task_id}"):
        filepath = to_ingest[index]
        logger.debug("processing %s", str(filepath).replace(str(DATA_DIR) + "/", ""))
        batch.process_image(filepath)

    # if we finish with an unwritten partial chunk, write it out
    if batch.count() > 0:
        batch.write()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse()
    with logging_redirect_tqdm():
        main(args)

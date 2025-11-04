"""
ingest.py
author: Connor Hainje

usage:
python ingest.py
"""

import logging

from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from talltable.batch import BatchWriter
from talltable.parallel_batch import ParallelBatchWriter

from talltable.query import get_image_filepaths
from talltable.paths import DATA_DIR


logger = logging.getLogger(__name__)


def parse():
    ap = ArgumentParser()
    ap.add_argument("mjd", nargs="?", default=None)
    ap.add_argument("-N", "--num-workers", type=int, nargs="?", default=8)
    ap.add_argument("-C", "--chunk-size", type=int, nargs="?", default=16)
    ap.add_argument("-f", "--force", action="store_true", help="force (re-ingest)")
    return ap.parse_args()


def main(args):
    data_files = glob(
        str(DATA_DIR / (("*" if args.mjd is None else f"{args.mjd}") + "/*.fits"))
    )
    done_before = set(get_image_filepaths()) if not args.force else set()
    to_ingest = set(data_files) - done_before
    to_ingest = sorted(list(to_ingest))
    logger.info("%d files to ingest", len(to_ingest))

    if args.num_workers > 1:
        batch = ParallelBatchWriter(num_workers=args.num_workers)

        for index in tqdm(range(0, len(to_ingest), args.chunk_size), unit='chunk'):
            filepaths = to_ingest[index:index+args.chunk_size]
            batch.process_batch(filepaths)

    else:
        batch = BatchWriter(chunk_size=args.chunk_size)

        for index in tqdm(range(len(to_ingest))):
            filepath = to_ingest[index]
            logger.info("processing %s", str(filepath).replace(str(DATA_DIR) + '/', ''))
            batch.process_image(filepath)

        # if we finish with an unwritten partial chunk, write it out
        if batch.count() > 0:
            batch.write()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse()
    with logging_redirect_tqdm():
        main(args)

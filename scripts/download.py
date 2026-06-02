"""
download.py
author: Connor Hainje

usage:
python download.py 60906 -N 8 --use-aws
"""

import boto3

from argparse import ArgumentParser
from astroquery.ipac.irsa import Irsa
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests import get, Session
from os.path import basename
from tqdm.auto import tqdm

from talltable.paths import require_env

DATA_DIR = require_env("TALLTABLE_DATA_DIR")


def parse():
    ap = ArgumentParser()
    ap.add_argument(
        "mjd",
        type=int,
        help="MJD of the day to download observations from",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="run the URL query but don't download the files",
    )
    ap.add_argument(
        "-N",
        "--num-workers",
        type=int,
        nargs="?",
        default=8,
        help="number of workers",
    )
    ap.add_argument(
        "--use-aws",
        action="store_true",
        help="download files from AWS instead of IRSA",
    )
    args = ap.parse_args()

    # validation
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    return args


def download_file(url, folder, session=None):
    filepath = folder / basename(url)
    if filepath.exists():
        return None
    response = session.get(url) if session is not None else get(url)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def download_file_from_s3(url, folder, s3):
    filepath = folder / basename(url)
    if filepath.exists():
        return None

    s3_key = url.split("spherex/")[1]
    s3.download_file("nasa-irsa-spherex", s3_key, filepath)
    return filepath


def main():
    args = parse()

    folder = DATA_DIR / f"{args.mjd}"
    folder.mkdir(exist_ok=True)

    query = f"""
        SELECT 'https://irsa.ipac.caltech.edu/' || a.uri as uri
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE FLOOR(p.time_bounds_lower) = {args.mjd}
    """
    print(f"executing query:\n{query}")
    urls = Irsa.query_tap(query).to_table()["uri"]
    print(f"got {len(urls)} image URLs for MJD = {args.mjd}")

    print("checking which urls are new...")
    new_urls = []
    for url in urls:
        filepath = folder / basename(url)
        if not filepath.exists():
            new_urls.append(url)
    print(f"done! {len(new_urls)} of {len(urls)} are new")

    if args.no_download:
        print("  --no-download specified: stopping!")
        return

    if args.use_aws:
        s3 = boto3.client("s3")
        print(f"starting downloads from S3 using {args.num_workers} workers")

        if args.num_workers == 1:
            for url in tqdm(new_urls):
                download_file_from_s3(url, folder, s3)
        else:
            session = Session()
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        download_file_from_s3,
                        url,
                        folder,
                        s3,
                    )
                    for url in new_urls
                ]
                for _ in tqdm(as_completed(futures), total=len(new_urls)):
                    pass

    else:
        print(f"starting downloads from IRSA using {args.num_workers} workers")

        if args.num_workers == 1:
            for url in tqdm(new_urls):
                download_file(url, folder)
        else:
            session = Session()
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        download_file,
                        url,
                        folder,
                        session,
                    )
                    for url in new_urls
                ]
                for _ in tqdm(as_completed(futures), total=len(new_urls)):
                    pass


if __name__ == "__main__":
    main()

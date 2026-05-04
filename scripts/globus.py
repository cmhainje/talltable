import globus_sdk
import globus_sdk.token_storage
import requests
import os
import time

from argparse import ArgumentParser
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from talltable.partition import (
    find_partitions_disc,
    find_partitions_rect,
    find_partitions_ipix,
)
from talltable.paths import PART_DB_PATH, IMAGE_DB_PATH, WAVES_DB_PATH, DB_DIR


CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
CLIENT_ID = "8b94fb22-2894-45ec-9cfd-ead4df1865b5"
DOMAIN = "https://g-f46d3f.e3fe4d.8443.data.globus.org"
COLLECTION_ID = "1ecf129f-97fa-4899-a3fd-822a668f53c6"
HTTPS_SCOPE = f"https://auth.globus.org/scopes/{COLLECTION_ID}/https"
TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"
TOKEN_FILE = os.path.expanduser("~/.talltable_globus_tokens.json")


def _do_login_flow(auth_client):
    auth_client.oauth2_start_flow(
        requested_scopes=[HTTPS_SCOPE, TRANSFER_SCOPE],
        refresh_tokens=True,
    )
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    auth_code = input("Please enter the code here: ").strip()
    return auth_client.oauth2_exchange_code_for_tokens(auth_code)


def get_headers():
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    storage = globus_sdk.token_storage.JSONTokenStorage(TOKEN_FILE)

    token_data = storage.get_token_data(COLLECTION_ID)

    if token_data is not None:
        # refresh if within 5 minutes of expiry
        if token_data.expires_at_seconds - time.time() < 300:
            resp = auth_client.oauth2_refresh_token(token_data.refresh_token)
            storage.store_token_response(resp)
            token_data = storage.get_token_data(COLLECTION_ID)
    else:
        tokens = _do_login_flow(auth_client)
        storage.store_token_response(tokens)
        token_data = storage.get_token_data(COLLECTION_ID)

    return {"Authorization": f"Bearer {token_data.access_token}"}


def is_valid_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def handle_disc(args):
    kw = dict(ra=args.ra, dec=args.dec, frame="icrs")
    if is_valid_float(args.ra):
        kw['unit'] = 'deg'

    x = SkyCoord(**kw)
    return find_partitions_disc(x.ra.deg, x.dec.deg, args.radius)


def handle_rect(args):
    kw = dict(ra=args.ra, dec=args.dec, frame="icrs")
    if is_valid_float(args.ra):
        kw['unit'] = 'deg'

    x = SkyCoord(**kw)
    return find_partitions_rect(x.ra.deg, x.dec.deg, args.width, args.height)


def handle_ipix(args):
    return find_partitions_ipix(args.indices, args.level)


def print_disc(args):
    print(f"selecting disc at {args.ra}, {args.dec} with radius {args.radius}")


def print_rect(args):
    print(
        f"selecting rectangle centered at {args.ra}, {args.dec} "
        + f"with width {args.width} and height {args.height}"
    )


def print_ipix(args):
    print(f"selecting healpix level {args.level} indices {args.indices}")


def parse():
    ap = ArgumentParser()
    subs = ap.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        dest="command",
        required=True,
    )

    ap_disc = subs.add_parser("disc", help="select a disc")
    ap_disc.add_argument("ra", help="central ra")
    ap_disc.add_argument("dec", help="central dec")
    ap_disc.add_argument("radius", type=float, help="radius of disc, in degrees")
    ap_disc.set_defaults(find_partitions=handle_disc)
    ap_disc.set_defaults(print=print_disc)

    ap_rect = subs.add_parser("rect", help="select a rectangle")
    ap_rect.add_argument("ra", help="central ra")
    ap_rect.add_argument("dec", help="central dec")
    ap_rect.add_argument("width", type=float, help="width of rectangle, in degrees")
    ap_rect.add_argument("height", type=float, help="height of rectangle, in degrees")
    ap_rect.set_defaults(find_partitions=handle_rect)
    ap_rect.set_defaults(print=print_rect)

    ap_ipix = subs.add_parser("hp", help="select a generic list of healpix indices")
    ap_ipix.add_argument("level", type=int, help="healpix resolution (Nside = 2^this)")
    ap_ipix.add_argument("indices", type=int, nargs="*", help="healpix indices")
    ap_ipix.set_defaults(find_partitions=handle_ipix)
    ap_ipix.set_defaults(print=print_ipix)

    args = ap.parse_args()
    return args


def download_file(path, dest_path, headers, prefix=""):
    url = f"{DOMAIN}/{path}"

    resp = requests.head(url, headers=headers, allow_redirects=False)
    if resp.is_redirect:
        raise RuntimeError(f"auth redirect for {url} — token may be invalid or missing required scope")

    expected_size = int(resp.headers["Content-Length"])

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            with tqdm(total=expected_size, unit="B", unit_scale=True, desc=f"{prefix}{path}") as bar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))

    actual_size = os.path.getsize(dest_path)
    if actual_size != expected_size:
        raise ValueError(f"size mismatch for {dest_path}: expected {expected_size}, got {actual_size}")


if __name__ == "__main__":
    # parse the arguments
    args = parse()
    args.print(args)

    # log in to globus, get headers
    headers = get_headers()
    print("login successful!")

    # copy over the auxilliary dbs (do this every time)
    download_file("parts.txt", PART_DB_PATH, headers)
    download_file("image.parquet", IMAGE_DB_PATH, headers)
    download_file("waves.parquet", WAVES_DB_PATH, headers)
    print("downloaded partition, image, and wavelength tables")

    # look up which partitions to copy and copy them
    partitions = args.find_partitions(args)
    print(f"the query region intersects {len(partitions)} files. downloading...")

    n = len(partitions)
    w = len(str(n))
    for i, p in enumerate(partitions, 1):
        _path = f"pixels/part={p}/compacted.parquet"
        download_file(_path, DB_DIR / _path, headers, prefix=f"({i:0{w}d}/{n}) ")

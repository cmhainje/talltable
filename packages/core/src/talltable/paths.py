import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# look for .env in the parents of the cwd *and* the parents of this file
# load both. loads *don't* overwrite, so precedence is
#     user env > cwd walk > package walk
load_dotenv(find_dotenv(usecwd=True))
load_dotenv()


def require_env(var: str) -> Path:
    val = os.environ.get(var)
    if val is None:
        raise RuntimeError(
            f"Environment variable {var!r} is not set.\n\n"
            "See example.env for the full list of configuration variables, or\n"
            "https://connorhainje.com/talltable for documentation."
        )
    return Path(val)


DB_DIR      = require_env("TALLTABLE_DB_DIR")
SCRATCH_DIR = Path(os.environ.get("TALLTABLE_SCRATCH_DIR", DB_DIR))

PIXEL_DB_PATH   = DB_DIR / "pixels"
IMAGE_DB_PATH   = DB_DIR / "image.parquet"
WAVES_DB_PATH   = DB_DIR / "waves.parquet"
EPHEM_DB_PATH   = DB_DIR / "ephem.parquet"
WCS_DB_PATH     = DB_DIR / "wcs.parquet"
IMAGE_PARTS_DIR = DB_DIR / "image_parts"
PART_DB_PATH    = DB_DIR / "parts.txt"


def image_part_path(task_id: int) -> Path:
    return IMAGE_PARTS_DIR / f"image_task{task_id}.parquet"

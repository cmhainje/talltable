import tomllib
from pathlib import Path

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _cfg = tomllib.load(f)["paths"]

DATA_DIR    = Path(_cfg["data_dir"])
DB_DIR      = Path(_cfg["db_dir"])
SCRATCH_DIR = Path(_cfg["scratch_dir"])

PIXEL_DB_PATH   = DB_DIR / "pixels"
IMAGE_DB_PATH   = DB_DIR / "image.parquet"
WAVES_DB_PATH   = DB_DIR / "waves.parquet"
EPHEM_DB_PATH   = DB_DIR / "ephem.parquet"
WCS_DB_PATH     = DB_DIR / "wcs.parquet"
IMAGE_PARTS_DIR = DB_DIR / "image_parts"
PART_DB_PATH    = DB_DIR / "parts.txt"


def image_part_path(task_id: int) -> Path:
    return IMAGE_PARTS_DIR / f"image_task{task_id}.parquet"

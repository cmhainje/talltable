from pathlib import Path

DATA_DIR = Path("/mnt/sdceph/users/spherex/spherex_data_qr2")
DB_DIR = Path("/mnt/sdceph/users/chainje/spxdb-test")

PIXEL_DB_PATH = DB_DIR / "pixels"
IMAGE_DB_PATH = DB_DIR / "image.parquet"
WAVES_DB_PATH = DB_DIR / "waves.parquet"
EPHEM_DB_PATH = DB_DIR / "ephem.parquet"
IMAGE_PARTS_DIR = DB_DIR / "image_parts"

PART_DB_PATH = DB_DIR / "parts.txt"


def image_part_path(task_id: int) -> Path:
    return IMAGE_PARTS_DIR / f"image_task{task_id}.parquet"

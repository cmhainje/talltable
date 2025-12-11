from pathlib import Path

SRC_DIR  = Path(__file__).parent.absolute()
PROJ_DIR = SRC_DIR.parent.absolute()

DATA_DIR = Path("/Volumes/databoy/spherex")

DB_DIR   = (PROJ_DIR / "db").absolute()
PIXEL_DB_PATH = DB_DIR / "pixels"
IMAGE_DB_PATH = DB_DIR / "image.parquet"
WAVES_DB_PATH = DB_DIR / "waves.parquet"
EPHEM_DB_PATH = DB_DIR / "ephem.parquet"

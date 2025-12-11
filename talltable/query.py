import duckdb

from .paths import IMAGE_DB_PATH, PIXEL_DB_PATH, WAVES_DB_PATH, EPHEM_DB_PATH


DUCK_IMAGE = "'" + str(IMAGE_DB_PATH) + "'"
DUCK_WAVES = "'" + str(WAVES_DB_PATH) + "'"
DUCK_EPHEM = "'" + str(EPHEM_DB_PATH) + "'"
DUCK_PIXEL = "'" + str(PIXEL_DB_PATH / "*/*.parquet") + "'"


def get_image_filepaths() -> list[str]:
    if IMAGE_DB_PATH.exists():
        query = duckdb.sql(f"SELECT filepath FROM {DUCK_IMAGE}")
        output = query.fetchnumpy()["filepath"]
        return [str(x) for x in output]
    else:
        return []

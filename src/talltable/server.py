import duckdb

from fastapi import FastAPI

from talltable.constants import HP_HIGH_LEVEL
from talltable.partition import find_partitions_rect
from talltable.paths import DB_DIR, WAVES_DB_PATH, PIXEL_DB_PATH


app = FastAPI()
con = duckdb.connect()

# load the wavelengths
con.execute(f"""
CREATE TABLE waves AS
    FROM read_parquet('{WAVES_DB_PATH}');
""")

# load the partitions
with open(DB_DIR / "parts.txt", "r") as f:
    all_parts = set(int(line.strip()) for line in f.readlines())


@app.get("/")
def status():
    return "NYU Talltable Web Server"


@app.get("/q/{ra}/{dec}")
def query(ra: float, dec: float, radius: float = 1):
    _ = {
        "ra": ra,
        "dec": dec,
        "radius": radius,
    }
    raise NotImplementedError()


@app.get("/map/{ra}/{dec}")
def map(
    ra: float,
    dec: float,
    width: float = 1,
    height: float = 1,
    maplvl: int = 6,
    wavemin: float = 1.85,
    wavemax: float = 1.90,
):
    """
    ra: degrees
    dec: degrees
    width: degrees
    height: degrees
    maplvl: healpix level of map
    """

    query_parts = find_partitions_rect(ra, dec, width, height, all_parts=all_parts)

    if len(query_parts) == 0:
        raise ValueError("no partitions found in the desired region")

    paths = [f"'{PIXEL_DB_PATH}/part={part}/compacted.parquet'" for part in query_parts]
    paths = ",".join(paths)

    query = f"""
    SELECT
        hphigh >> 2 * ({HP_HIGH_LEVEL} - {maplvl}) AS mappix,
        SUM( (flux - zodi) / variance )            AS numerator,
        SUM( 1.0 / variance )                      AS denominator,
    FROM read_parquet({paths}) p
    JOIN (
        SELECT waveid FROM waves
        WHERE wavelength BETWEEN {wavemin} AND {wavemax}
    ) w ON p.waveid = w.waveid
    GROUP BY mappix
    """

    return con.execute(query).fetchall()

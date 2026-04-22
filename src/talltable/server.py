import asyncio

import healpy as hp
import numpy as np
import duckdb

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from talltable.constants import HP_HIGH_LEVEL
from talltable.partition import find_partitions_rect, find_partitions_ipix8
from talltable.paths import DB_DIR, WAVES_DB_PATH, PIXEL_DB_PATH


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

con = duckdb.connect()

# load the wavelengths
con.execute(f"""
CREATE TABLE waves AS
    FROM read_parquet('{WAVES_DB_PATH}');
""")

# load the partitions
with open(DB_DIR / "parts.txt", "r") as f:
    all_parts = set(int(line.strip()) for line in f.readlines())


def _build_query(chunks, maplvl, wavemin, wavemax):
    query_parts = find_partitions_ipix8(chunks, all_parts=all_parts)
    if not query_parts:
        return None

    paths = [f"'{PIXEL_DB_PATH}/part={part}/compacted.parquet'" for part in query_parts]
    paths = ",".join(paths)
    if len(query_parts) > 1:
        paths = f"[{paths}]"

    chunk_list_sql = ",".join(str(int(c)) for c in chunks)
    return f"""
    SELECT
        (hphigh >> 28)                                    AS chunk_id,
        (hphigh >> (2 * ({HP_HIGH_LEVEL} - {maplvl})))   AS mappix,
        SUM((flux - zodi) / variance) / SUM(1.0 / variance) AS flux
    FROM read_parquet({paths}) p
    JOIN (SELECT waveid FROM waves
          WHERE wavelength BETWEEN {wavemin} AND {wavemax}) w
      ON p.waveid = w.waveid
    WHERE (hphigh >> 28) IN ({chunk_list_sql})
      AND flags & ~(1 << 21) = 0
    GROUP BY chunk_id, mappix
    ORDER BY chunk_id, mappix
    """


def _tile_msg(seq: int, is_final: bool, chunk_id: int, mappix_list, flux_list) -> bytes:
    """Binary message: [seq u32][flags u32][nTiles u32][chunkId u32][nPix u32][mappix u64...][flux f32...]"""
    flags = np.uint32(1 if is_final else 0).tobytes()
    nPix = len(mappix_list)
    return (
        np.uint32(seq).tobytes()
        + flags
        + np.uint32(1).tobytes()
        + np.uint32(chunk_id).tobytes()
        + np.uint32(nPix).tobytes()
        + np.array(mappix_list, dtype=np.uint64).tobytes()
        + np.array(flux_list, dtype=np.float32).tobytes()
    )


def _empty_final_msg(seq: int) -> bytes:
    return np.uint32(seq).tobytes() + np.uint32(1).tobytes() + np.uint32(0).tobytes()


def stream_tiles(chunks: list, maplvl: int, wavemin: float, wavemax: float, seq: int):
    """Generator yielding binary messages — one per HEALPix chunk, last one flagged isFinal."""
    if not chunks:
        yield _empty_final_msg(seq)
        return

    query = _build_query(chunks, maplvl, wavemin, wavemax)
    if query is None:
        yield _empty_final_msg(seq)
        return

    cursor = con.execute(query)

    current_chunk = None
    mappix_list = []
    flux_list = []
    last_rows = None

    while True:
        rows = cursor.fetchmany(10_000)
        if not rows:
            break
        for chunk_id, mappix, flux in rows:
            if chunk_id != current_chunk:
                if current_chunk is not None:
                    yield _tile_msg(seq, False, current_chunk, mappix_list, flux_list)
                current_chunk = chunk_id
                mappix_list = [mappix]
                flux_list = [flux]
            else:
                mappix_list.append(mappix)
                flux_list.append(flux)
    if current_chunk is not None:
        yield _tile_msg(seq, True, current_chunk, mappix_list, flux_list)
    else:
        yield _empty_final_msg(seq)


def compute_tiles_binary(
    chunks: list,
    maplvl: int = 6,
    wavemin: float = 1.85,
    wavemax: float = 1.90,
    seq: int = 0,
) -> bytes:
    """Non-streaming version kept for the REST endpoint."""
    if not chunks:
        return np.uint32(seq).tobytes() + np.uint32(0).tobytes()

    query = _build_query(chunks, maplvl, wavemin, wavemax)
    if query is None:
        return np.uint32(seq).tobytes() + np.uint32(0).tobytes()

    rows = con.execute(query).fetchall()
    if not rows:
        return np.uint32(seq).tobytes() + np.uint32(0).tobytes()

    tiles = []
    current_chunk = None
    mappix_list = []
    flux_list = []
    for chunk_id, mappix, flux in rows:
        if chunk_id != current_chunk:
            if current_chunk is not None:
                tiles.append((current_chunk, mappix_list, flux_list))
            current_chunk = chunk_id
            mappix_list = [mappix]
            flux_list = [flux]
        else:
            mappix_list.append(mappix)
            flux_list.append(flux)
    if current_chunk is not None:
        tiles.append((current_chunk, mappix_list, flux_list))

    buf = np.uint32(seq).tobytes() + np.uint32(len(tiles)).tobytes()
    for chunk_id, mappix_list, flux_list in tiles:
        nPix = len(mappix_list)
        buf += np.uint32(chunk_id).tobytes()
        buf += np.uint32(nPix).tobytes()
        buf += np.array(mappix_list, dtype=np.uint32).tobytes()
        buf += np.array(flux_list, dtype=np.float32).tobytes()
    return buf


@app.get("/")
def status():
    return "NYU Talltable Web Server"


@app.get("/q/{ra}/{dec}")
def query(ra: float, dec: float, radius: float = 1):
    raise NotImplementedError()


@app.get("/chunks")
def get_chunks(ra: float, dec: float, width: float, height: float):
    ipix = hp.query_polygon(
        256,  # Nside = 2^8
        hp.ang2vec(
            ra  + 0.5 * width  * np.array([-1, +1, +1, -1]),
            dec + 0.5 * height * np.array([-1, -1, +1, +1]),
            lonlat=True,
        ),
        nest=True,
        inclusive=True,
    )
    return {"chunks": sorted(int(i) for i in ipix)}


@app.get("/map/{ra}/{dec}")
def map_get(
    ra: float,
    dec: float,
    width: float = 1,
    height: float = 1,
    maplvl: int = 6,
    wavemin: float = 1.85,
    wavemax: float = 1.90,
):
    """REST endpoint kept for easy ad-hoc testing."""
    ipix = hp.query_polygon(
        256,  # Nside = 2^8
        hp.ang2vec(
            ra  + 0.5 * width  * np.array([-1, +1, +1, -1]),
            dec + 0.5 * height * np.array([-1, -1, +1, +1]),
            lonlat=True,
        ),
        nest=True,
        inclusive=True,
    )
    chunks = sorted(int(i) for i in ipix)
    buf = compute_tiles_binary(chunks=chunks, maplvl=maplvl, wavemin=wavemin, wavemax=wavemax, seq=0)
    return Response(content=buf, media_type="application/octet-stream")


@app.websocket("/ws/tiles")
async def ws_tiles(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            params = await websocket.receive_json()
            for buf in stream_tiles(**params):
                await websocket.send_bytes(buf)
                await asyncio.sleep(0)  # yield to event loop between chunks
    except WebSocketDisconnect:
        pass

import asyncio
import duckdb
import io
import json
import logging
import os
import pyarrow.ipc as ipc
import re
import secrets
import signal
import struct

from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from pyarrow import parquet as pq
from pydantic import BaseModel, model_validator
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError
from starlette.middleware.base import BaseHTTPMiddleware

from talltable.partition import (
    find_partitions_ipix,
    find_partitions_disc,
    find_partitions_rect,
)
from talltable.paths import DB_DIR, IMAGE_DB_PATH, WAVES_DB_PATH, PIXEL_DB_PATH, PART_DB_PATH

from . import extract


TMP_DB_PATH = Path("temp.duckdb")
TIMEOUT_SEC = 90.0
PSF_CALIB_DIR = Path(__file__).resolve().parents[2] / "calib" / "psf"
PARTS_RELOAD_SEC = float(os.environ.get("TALLTABLE_PARTS_RELOAD_SEC", 3600))
RESTART_TOKEN = os.environ.get("TALLTABLE_RESTART_TOKEN")
MAX_BODY_SIZE = 500 * 1024 * 1024  # 500 MB

FRAME_DATA = 0x00
FRAME_STATUS = 0x01

logger = logging.getLogger("uvicorn")


def _frame(data: bytes, frame_type: int = FRAME_DATA) -> bytes:
    return struct.pack(">IB", len(data), frame_type) + data


def _status_frame(ok: bool, message: str | None = None) -> bytes:
    payload = {"ok": ok}
    if message is not None:
        payload["message"] = message
    return _frame(json.dumps(payload).encode(), FRAME_STATUS)


def _read_all_parts() -> set[int]:
    with open(PART_DB_PATH, "r") as f:
        return set(int(line.strip()) for line in f.readlines())


ALL_PARTS = _read_all_parts()


async def _reload_all_parts_periodically():
    global ALL_PARTS
    while True:
        await asyncio.sleep(PARTS_RELOAD_SEC)
        try:
            new_parts = await asyncio.to_thread(_read_all_parts)
        except OSError:
            logger.exception("failed to reload parts.txt; keeping previous ALL_PARTS")
            continue
        ALL_PARTS = new_parts
        logger.info(f"reloaded ALL_PARTS: {len(ALL_PARTS)} partitions")


def _con_config():
    return {
        "threads": int(os.environ.get("DUCKDB_THREADS", 16)),
        "memory_limit": os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        "autoload_known_extensions": False,
        "autoinstall_known_extensions": False,
        "allow_community_extensions": False,
    }


# keep one dummy connection open to ensure that allowed_directories 
# and enable_external_access stay locked to current values
# (inherited by future connections)
_lockdown_con: duckdb.DuckDBPyConnection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lockdown_con

    # startup
    TMP_DB_PATH.unlink(missing_ok=True)
    TMP_DB_PATH.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    with duckdb.connect(TMP_DB_PATH) as con:
        con.execute(f"CREATE TABLE waves AS FROM read_parquet('{WAVES_DB_PATH}')")
        con.execute(f"CREATE TABLE images AS FROM read_parquet('{IMAGE_DB_PATH}')")

    extract.image_tab = pq.read_table(IMAGE_DB_PATH)
    extract.psf_cubes = [
        fits.open(PSF_CALIB_DIR / f"average_psf_D{det}_spx_cal-psf-v5-2026-082.fits")
        for det in range(1, 7)
    ]
    extract.sapm_images = None

    _lockdown_con = duckdb.connect(TMP_DB_PATH, read_only=True, config=_con_config())
    _lockdown_con.execute(f"SET allowed_directories=['{DB_DIR}']")
    _lockdown_con.execute("SET enable_external_access=false")
    _lockdown_con.execute("SET lock_configuration=true")

    reload_task = asyncio.create_task(_reload_all_parts_periodically())

    yield

    # teardown
    reload_task.cancel()
    try:
        await reload_task
    except asyncio.CancelledError:
        pass

    _lockdown_con.close()

    TMP_DB_PATH.unlink(missing_ok=True)
    TMP_DB_PATH.with_suffix(".duckdb.wal").unlink(missing_ok=True)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413, content={"detail": "Request body too large"}
            )
        return await call_next(request)


app = FastAPI(lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware)
executor = ThreadPoolExecutor(max_workers=10)


@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    request, exc
    return JSONResponse(
        status_code=500, content={"detail": "Request exceeded memory limits"}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request
    return JSONResponse(
        status_code=422, content={"detail": f"Request could not be processed. Error message:\n{exc}"}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # fallback for anything not caught by a more specific handler above/below
    # otherwise, the default handler would return a bare "Internal Server Error"
    request
    return JSONResponse(
        status_code=500, content={"detail": str(exc)}
    )


class PartitionUnavailable(Exception):
    """A known partition's file is missing because it's mid-ingest (compacting or split)."""
    def __init__(self, part: int):
        self.part = part
        super().__init__(
            f"partition {part} is temporarily unavailable (a database update is in "
            "progress); please retry shortly"
        )


@app.exception_handler(PartitionUnavailable)
async def partition_unavailable_handler(request: Request, exc: PartitionUnavailable):
    request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class ServerBusy(Exception):
    """No pixel query slot became free in time."""
    def __init__(self):
        super().__init__("the server is busy with other queries; please retry shortly")


@app.exception_handler(ServerBusy)
async def server_busy_handler(request: Request, exc: ServerBusy):
    request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def get_con():
    # inherits the lockdown already applied on _lockdown_con — see its comment
    return duckdb.connect(TMP_DB_PATH, read_only=True, config=_con_config())


# *** STATUS ***


@app.get("/")
@app.head("/")
def status():
    return {"service": "talltable simple web service", "status": "running ok!"}


@app.post("/restart")
async def restart(authorization: str | None = Header(default=None)):
    scheme, _, token = (authorization or "").partition(" ")
    if (
        not RESTART_TOKEN
        or scheme.lower() != "bearer"
        or not token
        or not secrets.compare_digest(token, RESTART_TOKEN)
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    async def _delayed_terminate():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_delayed_terminate())
    return {"ok": True, "message": "restarting"}


# *** SQL ***


# Pixel queries run (at most) this many at a time; more and they slow each other down until all of
# them time out and the server looks down to everyone. A slot is held for the whole query, from
# checking its partitions to the end of its result stream; the rest wait their turn, but no longer
# than SLOT_WAIT_SEC (a waiting request sends nothing, and the load balancer in front drops idle
# connections after 60 s), then get a 503 to retry.
query_slots = asyncio.Semaphore(int(os.environ.get("TALLTABLE_MAX_PIXEL_QUERIES", 2)))
SLOT_WAIT_SEC = 45.0


class QuerySlot:
    """A pixel query slot, given back exactly once: by release() when the query's stream ends, or
    else when this is garbage collected (a response whose stream never started, e.g. because the
    client left first, never runs the stream's cleanup)."""

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._held = True

    def release(self):
        if self._held:
            self._held = False
            try:
                self._loop.call_soon_threadsafe(query_slots.release)
            except RuntimeError:  # event loop closed: shutting down
                pass

    def __del__(self):
        self.release()


async def acquire_query_slot() -> QuerySlot:
    try:
        await asyncio.wait_for(query_slots.acquire(), timeout=SLOT_WAIT_SEC)
    except asyncio.TimeoutError:
        raise ServerBusy()
    return QuerySlot()


def check_partition(part, path):
    """A pixel query's partition file must exist: if it doesn't, the partition is either known but
    mid-split (503, retry) or not a partition at all (422).

    (This used to read the whole file first, to warm it. Since PixelQuery's filters let DuckDB skip
    the row groups a query doesn't need, that only cost time: ~4 s for a 2' cone that runs in ~1 s
    without it. Measured 2026-10-16.)"""
    if not path.exists():
        if part in ALL_PARTS:
            raise PartitionUnavailable(part)
        raise ValueError(f"partition file not found ({part})")


async def check_partitions(parts_paths):
    await asyncio.to_thread(lambda: [check_partition(p, path) for p, path in parts_paths])


def parse_sql(sql):
    try:
        return parse_one(sql, read="duckdb")
    except ParseError as e:
        raise ValueError(str(e))


# besides `pixels` (all of the request's partitions), a query may read `pixels_<part>`: just that one
# of the request's partitions. PixelQuery uses it to give each partition its own hphigh filter.
PIXELS_PART_TABLE = re.compile(r"pixels_(\d+)")


def uses_pixels(sql):
    for table in parse_sql(sql).find_all(exp.Table):
        if table.name == "pixels" or PIXELS_PART_TABLE.fullmatch(table.name):
            return True
    return False


class SQLRequest(BaseModel):
    query: str
    partitions: list[int] | None = None

    @model_validator(mode="after")
    def partitions_required_for_pixels(self) -> "SQLRequest":
        if uses_pixels(self.query) and not self.partitions:
            raise ValueError("partitions must be provided when query contains 'pixels'")
        return self

    def into_sql(self) -> str:
        if not uses_pixels(self.query):
            return self.query

        def part_path(p):
            return str(PIXEL_DB_PATH / f"part={p}/compacted.parquet")

        partition_sql = ",".join([f"'{part_path(p)}'" for p in self.partitions])
        if len(self.partitions) > 1:
            partition_sql = f"[{partition_sql}]"

        def transformer(node):
            if not isinstance(node, exp.Table):
                return node
            if node.name == "pixels":
                repl = parse_sql(f"read_parquet({partition_sql})")
            elif m := PIXELS_PART_TABLE.fullmatch(node.name):
                part = int(m[1])
                if part not in self.partitions:
                    raise ValueError(f"{node.name}: partition {part} is not among the request's partitions")
                repl = parse_sql(f"read_parquet('{part_path(part)}')")
            else:
                return node
            alias = node.args.get("alias")
            if alias:
                return exp.Alias(this=repl, alias=alias)
            return repl

        return (
            parse_sql(self.query)
            .transform(transformer)
            .sql(dialect="duckdb")
        )


def stream_with_timeout(
    gen, timeout_chunk=None, on_timeout=None, on_done=None
):
    """`on_done` runs however the stream ends: completed, timed out, failed, or abandoned by a
    client that disconnected (the response then closes this generator; `on_timeout` is called
    for that too, so the query stops rather than running on for nobody)."""
    async def async_generator():
        loop = asyncio.get_event_loop()
        deadline = loop.time() + TIMEOUT_SEC
        stopped = False  # finished, or already interrupted on timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(executor, next, gen),
                        timeout=remaining,
                    )
                    yield chunk
                except (StopIteration, RuntimeError) as e:
                    if isinstance(e, RuntimeError) and "StopIteration" not in str(e):
                        raise
                    stopped = True
                    break
        except asyncio.TimeoutError:
            stopped = True
            if on_timeout is not None:
                on_timeout()
            # Don't call gen.close() — the thread is still running next(gen) and
            # calling close() on a running generator raises ValueError. on_timeout()
            # (i.e. con.interrupt()) unblocks it; the generator's finally will fire
            # naturally once the thread returns.
            if timeout_chunk is not None:
                yield timeout_chunk
        finally:
            if not stopped and on_timeout is not None:
                try:
                    on_timeout()
                except Exception:  # e.g. the query already failed and closed its connection
                    pass
            if on_done is not None:
                on_done()

    return async_generator()


def arrow_stream(
    sql: str, con: duckdb.DuckDBPyConnection, params=None, batch_rows: int = 10_000
):
    try:
        reader = con.execute(sql, params or []).fetch_record_batch(batch_rows)
        sink = io.BytesIO()
        writer = ipc.new_stream(sink, reader.schema)
        for batch in reader:
            writer.write_batch(batch)
            data = sink.getvalue()
            if data:
                yield _frame(data)
            sink.seek(0)
            sink.truncate(0)
        writer.close()
        tail = sink.getvalue()
        if tail:
            yield _frame(tail)
        yield _status_frame(True)
    except duckdb.Error as e:
        yield _status_frame(False, str(e))
    finally:
        con.close()


async def start_query(req: SQLRequest):
    """Take a pixel query slot (if it's a pixel query) and get the query ready to stream: returns
    (the query, a connection, the function that gives the slot back, or None)."""
    slot = await acquire_query_slot() if req.partitions is not None else None
    try:
        if req.partitions is not None:
            parts_paths = [
                (p, PIXEL_DB_PATH / f'part={p}/compacted.parquet') for p in req.partitions
            ]
            await check_partitions(parts_paths)
        query = req.into_sql()
        con = get_con()
    except BaseException:
        if slot is not None:
            slot.release()
        raise
    return query, con, (slot.release if slot is not None else None)


@app.post("/sql")
async def sql(req: SQLRequest):
    query, con, release = await start_query(req)
    return StreamingResponse(
        stream_with_timeout(
            arrow_stream(query, con),
            timeout_chunk=_status_frame(False, "Query timed out"),
            on_timeout=con.interrupt,
            on_done=release,
        ),
        media_type="application/vnd.apache.arrow.stream",
    )


def json_stream(
    sql: str, con: duckdb.DuckDBPyConnection, params=None, batch_rows: int = 10_000
):
    yield '{"batches":['
    try:
        reader = con.execute(sql, params or []).fetch_record_batch(batch_rows)
        col_names = reader.schema.names

        first_batch = True
        for batch in reader:
            parts = {
                name: batch.column(i).to_pylist()
                for i, name in enumerate(col_names)
            }
            batch_str = json.dumps(parts, default=str)
            if not first_batch:
                batch_str = "," + batch_str
            first_batch = False
            yield batch_str

        yield '], "message": null}'
    except duckdb.Error as e:
        yield f'], "message": {json.dumps(str(e))}}}'
    finally:
        con.close()


@app.post("/sql.json")
async def sql_json(req: SQLRequest):
    query, con, release = await start_query(req)
    return StreamingResponse(
        stream_with_timeout(
            json_stream(query, con),
            timeout_chunk=f'], "message": {json.dumps("Query timed out")}}}',
            on_timeout=con.interrupt,
            on_done=release,
        ),
        media_type="application/json",
    )


# *** PARTITIONS ***


class PartitionsRequest(BaseModel):
    level: int
    indices: list[int]


@app.post("/partitions")
def partitions(req: PartitionsRequest):
    return {
        "partitions": list(
            map(int, find_partitions_ipix(req.indices, req.level, all_parts=ALL_PARTS))
        )
    }


@app.get("/partitions/disc")
def partitions_disc(ra: float, dec: float, radius: float):
    return {
        "partitions": list(
            map(int, find_partitions_disc(ra, dec, radius, all_parts=ALL_PARTS))
        )
    }


@app.get("/partitions/rect")
def partitions_rect(ra: float, dec: float, width: float, height: float):
    return {
        "partitions": list(
            map(int, find_partitions_rect(ra, dec, width, height, all_parts=ALL_PARTS))
        )
    }


# *** PHOTOMETRY ***


def _check_parts(parts):
    for p in parts:
        check_partition(p, PIXEL_DB_PATH / f"part={p}/compacted.parquet")


class FakeArgs:
    cutout_size = 0.05
    nchunks = 2
    auto_deblend = None
    deblend_radius = 20.0
    deblend_maglim = None
    fit_radius = 4.0
    kappa = 4.0
    max_iter = 10
    linear_bkg = False
    debug = False
    show_figs = False
    save_figs = False
    results_dir = "spherex_results"
    no_masking = False
    quiet = True


class PhotometryRequest(BaseModel):
    ra: float
    dec: float
    pm_ra: float | None = None
    pm_dec: float | None = None
    ref_epoch: float = 57388.0


@app.get("/photometry")
def photometry(ra: float, dec: float, pm_ra: float | None = None, pm_dec: float | None = None, ref_epoch: float = 57388.0):
    con = get_con()
    try:
        results = extract.process_target("_", ra, dec, pm_ra, pm_dec, ref_epoch, FakeArgs(), con=con, warm_fn=_check_parts)
    finally:
        con.close()
    return results


# class PhotometryBatchRequest(BaseModel):
#     batch: list[PhotometryRequest]
# 
# 
# @app.post("/photometry/batch")
# def batch_photometry(batch: PhotometryBatchRequest):
#     all_results = []
#     con = get_con()
#     try:
#         for r in batch.batch:
#             results = extract.process_target(
#                 "_", r.ra, r.dec,
#                 r.pm_ra, r.pm_dec, r.ref_epoch,
#                 FakeArgs(),
#                 con=con,
#                 warm_fn=_check_parts,
#             )
#             all_results.append(results)
#     finally:
#         con.close()
# 
#     return all_results



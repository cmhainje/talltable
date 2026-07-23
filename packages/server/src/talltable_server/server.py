import asyncio
import duckdb
import io
import json
import logging
import os
import pyarrow.ipc as ipc
import secrets
import signal
import struct

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
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


TMP_DB_PATH = Path("temp.duckdb")
TIMEOUT_SEC = 90.0
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


prefetch_sem = asyncio.Semaphore(2)

def warm(part, path):
    try:
        with open(path, 'rb') as f:
            while f.read(4*1024*1024):
                pass
    except OSError as e:
        if part in ALL_PARTS:
            raise PartitionUnavailable(part) from e
        raise ValueError(f"partition file not found ({part})") from e

async def prefetch_files(parts_paths, max_workers=16):
    async with prefetch_sem:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            await asyncio.gather(
                *[loop.run_in_executor(ex, warm, p, path) for p, path in parts_paths]
            )


def parse_sql(sql):
    try:
        return parse_one(sql, read="duckdb")
    except ParseError as e:
        raise ValueError(str(e))


def uses_pixels(sql):
    for table in parse_sql(sql).find_all(exp.Table):
        if table.name == "pixels":
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
            if isinstance(node, exp.Table) and node.name == "pixels":
                repl = parse_sql(f"read_parquet({partition_sql})")
                alias = node.args.get("alias")
                if alias:
                    return exp.Alias(this=repl, alias=alias)
                return repl
            return node

        return (
            parse_sql(self.query)
            .transform(transformer)
            .sql(dialect="duckdb")
        )


def stream_with_timeout(
    gen, timeout_chunk=None, on_timeout=None
):
    async def async_generator():
        loop = asyncio.get_event_loop()
        deadline = loop.time() + TIMEOUT_SEC
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
                    break
        except asyncio.TimeoutError:
            if on_timeout is not None:
                on_timeout()
            # Don't call gen.close() — the thread is still running next(gen) and
            # calling close() on a running generator raises ValueError. on_timeout()
            # (i.e. con.interrupt()) unblocks it; the generator's finally will fire
            # naturally once the thread returns.
            if timeout_chunk is not None:
                yield timeout_chunk

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


@app.post("/sql")
async def sql(req: SQLRequest):
    if req.partitions is not None:
        parts_paths = [
            (p, PIXEL_DB_PATH / f'part={p}/compacted.parquet') for p in req.partitions
        ]
        await prefetch_files(parts_paths)

    con = get_con()
    return StreamingResponse(
        stream_with_timeout(
            arrow_stream(req.into_sql(), con),
            timeout_chunk=_status_frame(False, "Query timed out"),
            on_timeout=con.interrupt,
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
    if req.partitions is not None:
        parts_paths = [
            (p, PIXEL_DB_PATH / f'part={p}/compacted.parquet') for p in req.partitions
        ]
        await prefetch_files(parts_paths)

    con = get_con()
    return StreamingResponse(
        stream_with_timeout(
            json_stream(req.into_sql(), con),
            timeout_chunk=f'], "message": {json.dumps("Query timed out")}}}',
            on_timeout=con.interrupt,
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

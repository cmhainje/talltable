import asyncio
import duckdb
import io
import json
import os
import pyarrow.ipc as ipc
import struct

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel, model_validator
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError

from talltable.partition import (
    find_partitions_ipix,
    find_partitions_disc,
    find_partitions_rect,
)
from talltable.paths import IMAGE_DB_PATH, WAVES_DB_PATH, PIXEL_DB_PATH, PART_DB_PATH


TMP_DB_PATH = Path("temp.duckdb")
TIMEOUT_SEC = 90.0

FRAME_DATA = 0x00
FRAME_STATUS = 0x01


def _frame(data: bytes, frame_type: int = FRAME_DATA) -> bytes:
    return struct.pack(">IB", len(data), frame_type) + data


def _status_frame(ok: bool, message: str | None = None) -> bytes:
    payload = {"ok": ok}
    if message is not None:
        payload["message"] = message
    return _frame(json.dumps(payload).encode(), FRAME_STATUS)


with open(PART_DB_PATH, "r") as f:
    ALL_PARTS = set(int(line.strip()) for line in f.readlines())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    TMP_DB_PATH.unlink(missing_ok=True)
    TMP_DB_PATH.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    with duckdb.connect(TMP_DB_PATH) as con:
        con.execute(f"CREATE TABLE waves AS FROM read_parquet('{WAVES_DB_PATH}')")
        con.execute(f"CREATE TABLE images AS FROM read_parquet('{IMAGE_DB_PATH}')")

    yield

    # teardown
    TMP_DB_PATH.unlink(missing_ok=True)
    TMP_DB_PATH.with_suffix(".duckdb.wal").unlink(missing_ok=True)


app = FastAPI(lifespan=lifespan)
executor = ThreadPoolExecutor(max_workers=10)


@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    request, exc
    return JSONResponse(
        status_code=500, content={"detail": "Request exceeded memory limits"}
    )


def get_con():
    return duckdb.connect(
        TMP_DB_PATH,
        read_only=True,
        config={
            "threads": int(os.environ.get("DUCKDB_THREADS", 16)),
            "memory_limit": os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        },
    )


# *** STATUS ***


@app.get("/")
@app.head("/")
def status():
    return {"service": "talltable simple web service", "status": "running ok!"}


# *** SQL ***


prefetch_sem = asyncio.Semaphore(2)

def warm(path):
    with open(path, 'rb') as f:
        while f.read(4*1024*1024):
            pass

async def prefetch_files(paths, max_workers=16):
    async with prefetch_sem:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            await asyncio.gather(*[loop.run_in_executor(ex, warm, p) for p in paths])


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
        if uses_pixels(self.query) and self.partitions is None:
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
        try:
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
    except duckdb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.post("/sql")
async def sql(req: SQLRequest):
    con = get_con()

    if req.partitions is not None:
        paths = [PIXEL_DB_PATH / f'part={p}/compacted.parquet' for p in req.partitions]
        await prefetch_files(paths)

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
    try:
        reader = con.execute(sql, params or []).fetch_record_batch(batch_rows)
        col_names = reader.schema.names

        yield '{"batches":['

        first_batch = True
        try:
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

    except duckdb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.post("/sql.json")
async def sql_json(req: SQLRequest):
    con = get_con()

    if req.partitions is not None:
        paths = [PIXEL_DB_PATH / f'part={p}/compacted.parquet' for p in req.partitions]
        await prefetch_files(paths)

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

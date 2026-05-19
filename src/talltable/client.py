import io
import json
import requests
import struct
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from pathlib import Path

from talltable.constants import SERVICE_URL

BASE_URL = SERVICE_URL

FRAME_DATA   = 0x00
FRAME_STATUS = 0x01


def _unframe(raw: bytes) -> tuple[bytes, dict]:
    """Split framed response into raw Arrow IPC bytes and the status dict."""
    buf = io.BytesIO(raw)
    arrow_buf = io.BytesIO()
    status = None

    while True:
        header = buf.read(5)
        if not header:
            break
        length, frame_type = struct.unpack('>IB', header)
        payload = buf.read(length)
        if frame_type == FRAME_DATA:
            arrow_buf.write(payload)
        elif frame_type == FRAME_STATUS:
            status = json.loads(payload)

    if status is None:
        raise RuntimeError("Server closed stream without a status frame (connection dropped?)")
    if not status["ok"]:
        raise RuntimeError(f"Query failed: {status.get('message', 'unknown error')}")

    arrow_buf.seek(0)
    return arrow_buf


def fetch_arrow_stream(url, payload):
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        raw = b"".join(response.iter_content(chunk_size=None))
        return ipc.open_stream(_unframe(raw)).read_all()

def fetch_arrow_to_parquet(url, payload, output):
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        arrow_buf = _unframe(response.content)
        reader = ipc.open_stream(arrow_buf)

        writer = None
        try:
            for batch in reader:
                if writer is None:
                    writer = pq.ParquetWriter(output, batch.schema)
                writer.write_batch(batch)
        finally:
            if writer:
                writer.close()

def sql_query(query: str, partitions: list[int], output_path: Path | None = None):
    if output_path is None:
        return fetch_arrow_stream(
            f"{BASE_URL}/sql",
            {
                "query": query,
                "partitions": partitions
            }
        )
    else:
        return fetch_arrow_to_parquet(
            f"{BASE_URL}/sql",
            {
                "query": query,
                "partitions": partitions
            },
            output_path
        )




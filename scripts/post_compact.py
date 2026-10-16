import logging
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from talltable.paths import (
    COMPACTION_ROUND_PATH,
    DB_DIR,
    IMAGE_DB_PATH,
    IMAGE_PARTS_DIR,
    PART_DB_PATH,
    PIXEL_DB_PATH,
    SCRATCH_DIR,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def cleanup_split_parents():
    """For partitions that were split, remove old compacted.parquet and .split marker."""
    markers = sorted(PIXEL_DB_PATH.glob("**/.split"))
    logger.info(f"cleaning up {len(markers)} split parents")
    for marker in markers:
        part_dir = marker.parent
        old_pq = part_dir / "compacted.parquet"
        if old_pq.exists():
            old_pq.unlink()
        marker.unlink()
        # remove dir if empty
        if part_dir.exists() and not any(part_dir.iterdir()):
            part_dir.rmdir()


def merge_image_parts():
    """Merge new partial image parquet files into the final images table."""
    part_files = sorted(IMAGE_PARTS_DIR.glob("image_task*.parquet"))
    if not part_files:
        logger.info("no image part files to merge")
        return

    logger.info(f"merging {len(part_files)} image part files")
    tables = []
    if IMAGE_DB_PATH.exists():
        tables.append(pq.read_table(IMAGE_DB_PATH))
    for f in part_files:
        tables.append(pq.read_table(f))

    merged = pa.concat_tables(tables)
    tmp_file = Path(str(IMAGE_DB_PATH) + ".tmp")
    pq.write_table(merged, tmp_file)
    tmp_file.replace(IMAGE_DB_PATH)

    for f in part_files:
        f.unlink()
    logger.info(f"merged into {IMAGE_DB_PATH} ({len(merged)} rows)")


def delete_bin_chunks():
    """Delete all binary chunk files."""
    chunks = list(SCRATCH_DIR.glob("chunk_*.bin"))
    for c in chunks:
        c.unlink()
    logger.info(f"deleted {len(chunks)} binary chunks")


def end_compaction_round():
    """Remove compact.py's round file. Only once the round's chunk files are gone: until then, a
    re-run of compact.py must find the same round, so that it skips the partitions it already did."""
    COMPACTION_ROUND_PATH.unlink(missing_ok=True)
    logger.info(f"removed {COMPACTION_ROUND_PATH}")


def write_parts_list():
    """Write the list of partition directories."""
    partitions = sorted(
        [p.stem.split("=")[1] for p in PIXEL_DB_PATH.glob("part=*")], key=int
    )
    tmp_path = DB_DIR / "parts-tmp.txt"
    with open(tmp_path, "w") as f:
        f.write("\n".join(p for p in partitions))
    tmp_path.replace(PART_DB_PATH)
    logger.info(f"found {len(partitions)} partitions, wrote to {PART_DB_PATH}")


def cleanup_partial_writes():
    """Remove outputs a killed compaction left half-written (compact.py's WRITING_NAME), and the
    partition directories that leaves empty (a split child that was never completed)."""
    partial = sorted(PIXEL_DB_PATH.glob("*/compacted_writing.parquet"))
    logger.info(f"removing {len(partial)} partially written files")
    for f in partial:
        f.unlink()
        if not any(f.parent.iterdir()):
            f.parent.rmdir()


def main():
    # (compact.py replaces each partition's compacted.parquet in place; only splits are left to finish)
    cleanup_split_parents()
    cleanup_partial_writes()
    merge_image_parts()
    delete_bin_chunks()
    end_compaction_round()
    write_parts_list()


if __name__ == "__main__":
    main()

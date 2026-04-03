import logging
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from talltable.paths import PIXEL_DB_PATH, IMAGE_DB_PATH, IMAGE_PARTS_DIR, PART_DB_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def promote_staging_files():
    """Rename all compacted_new.parquet -> compacted.parquet (atomic per file)."""
    staging_files = sorted(PIXEL_DB_PATH.glob("**/compacted_new.parquet"))
    logger.info(f"promoting {len(staging_files)} staging files")
    for f in staging_files:
        target = f.parent / "compacted.parquet"
        f.replace(target)


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
    chunks = list(PIXEL_DB_PATH.glob("chunk_*.bin"))
    for c in chunks:
        c.unlink()
    logger.info(f"deleted {len(chunks)} binary chunks")


def write_parts_list():
    """Write the list of partition directories."""
    partitions = sorted(
        [p.stem.split("=")[1] for p in PIXEL_DB_PATH.glob("part=*")], key=int
    )
    with open(PART_DB_PATH, "w") as f:
        f.write("\n".join(p for p in partitions))
    logger.info(f"found {len(partitions)} partitions, wrote to {PART_DB_PATH}")


def main():
    promote_staging_files()
    cleanup_split_parents()
    merge_image_parts()
    delete_bin_chunks()
    write_parts_list()


if __name__ == "__main__":
    main()

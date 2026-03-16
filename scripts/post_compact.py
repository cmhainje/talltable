from talltable.paths import PART_DB_PATH, PIXEL_DB_PATH, SCRATCH_DIR

partitions = sorted(
    [p.stem.split("=")[1] for p in PIXEL_DB_PATH.glob("part=*")], key=int
)
with open(PART_DB_PATH, "w") as f:
    f.write("\n".join(p for p in partitions))

print(f"found {len(partitions)} partitions, wrote to {PART_DB_PATH}")

leftover_chunks = list(SCRATCH_DIR.glob("chunk_*.hdf5"))
for p in leftover_chunks:
    p.unlink()
SCRATCH_DIR.rmdir()

print(f"deleted {len(leftover_chunks)} chunks")

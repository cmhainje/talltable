from talltable.paths import PART_DB_PATH, PIXEL_DB_PATH

partitions = sorted(
    [p.stem.split("=")[1] for p in PIXEL_DB_PATH.glob("part=*")], key=int
)

with open(PART_DB_PATH, "w") as f:
    f.write("\n".join(p for p in partitions))

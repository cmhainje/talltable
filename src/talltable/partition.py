import numpy as np
import healpy as hp
from .constants import PART_MIN_LEVEL, PART_MAX_LEVEL
from .paths import DB_DIR



def part_to_level_index(part):
    if isinstance(part, int):
        n = part.bit_length() - 1
    elif isinstance(part, np.ndarray):
        n = np.frexp(part)[1] - 1
    else:
        raise ValueError(f"unsupported argument type: {type(part)}")

    level = (n // 2) - 4
    index = part - (1 << n)
    return level, index


def level_index_to_part(level, index):
    return index + (1 << (2 * (level + 4)))


def find_partition(ra, dec):
    part = level_index_to_part(
        PART_MAX_LEVEL,
        hp.ang2pix(2**PART_MAX_LEVEL, ra, dec, nest=True, lonlat=True),
    )

    for _ in range(PART_MAX_LEVEL - PART_MIN_LEVEL + 1):
        if (DB_DIR / f"part={part}/compacted.parquet").exists():
            break
        part = part >> 2
    else:
        return None

    return part

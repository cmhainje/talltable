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


def find_partitions_ipix(ipix, level=PART_MAX_LEVEL, all_parts=None):
    if all_parts is None:
        with open(DB_DIR / "parts.txt", "r") as f:
            all_parts = set(int(line.strip()) for line in f.readlines())

    if level < PART_MAX_LEVEL:
        raise ValueError(f"level must be >= {PART_MAX_LEVEL}")

    ipix = np.atleast_1d(ipix)
    if level > PART_MAX_LEVEL:
        ipix = ipix >> (2 * (level - PART_MAX_LEVEL))

    query_parts = level_index_to_part(PART_MAX_LEVEL, ipix)
    query_parts = (
        set(query_parts)
        | set(query_parts >> 2)
        | set(query_parts >> 4)
        | set(query_parts >> 6)
        | set(query_parts >> 8)
    ) & all_parts

    return query_parts


def find_partitions_disc(ra, dec, radius, all_parts=None):
    ipix = hp.query_disc(
        2**PART_MAX_LEVEL,
        hp.ang2vec(ra, dec, lonlat=True),
        np.radians(radius),
        nest=True,
        inclusive=True,
    )
    return find_partitions_ipix(ipix, PART_MAX_LEVEL, all_parts=all_parts)


def find_partitions_rect(ra, dec, width, height, all_parts=None):
    ipix = hp.query_polygon(
        2**PART_MAX_LEVEL,
        hp.ang2vec(
            ra + 0.5 * width * np.array([-1, +1, +1, -1]),
            dec + 0.5 * height * np.array([-1, -1, +1, +1]),
            lonlat=True,
        ),
        nest=True,
        inclusive=True,
    )
    return find_partitions_ipix(ipix, PART_MAX_LEVEL, all_parts=all_parts)


def find_partitions_ipix8(ipix8, all_parts=None):
    """Find DB partitions overlapping the given level-8 HEALPix NESTED indices."""
    ipix8 = np.atleast_1d(np.asarray(ipix8, dtype=np.int64))
    # Each level-8 pixel has 4^2=16 level-10 children
    ipix10 = ((ipix8[:, None] << 4) | np.arange(16, dtype=np.int64)).ravel()
    return find_partitions_ipix(ipix10, level=PART_MAX_LEVEL, all_parts=all_parts)

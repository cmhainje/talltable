import duckdb
import numpy as np

from .constants import PART_MAX_LEVEL, HP_HIGH_LEVEL
from .paths import IMAGE_DB_PATH, PIXEL_DB_PATH, WAVES_DB_PATH, EPHEM_DB_PATH


DUCK_IMAGE = "'" + str(IMAGE_DB_PATH) + "'"
DUCK_WAVES = "'" + str(WAVES_DB_PATH) + "'"
DUCK_EPHEM = "'" + str(EPHEM_DB_PATH) + "'"
DUCK_PIXEL = "'" + str(PIXEL_DB_PATH / "**/*.parquet") + "'"


def choose_filter_level(region_area_deg2: float, target_pixels: int = 1000) -> int:
    """
    choose a HEALPix level L such that the region spans roughly `target_pixels` pixels at that level.

    uses the approximation N_pix ≈ 3 * area_rad^2 * 4^L, solves for L, then clamps to [PART_MAX_LEVEL, HP_HIGH_LEVEL]
    """
    area_rad2 = (np.pi / 180) ** 2 * region_area_deg2
    nside_sq = target_pixels * np.pi / (3.0 * area_rad2)
    L = round(np.log2(nside_sq) / 2)
    return max(PART_MAX_LEVEL, min(HP_HIGH_LEVEL, L))


def indices_to_hphigh_ranges(
    indices: np.ndarray | list[int],
    level: int,
) -> list[tuple[int, int]]:
    """
    convert HEALPix indices at `level` to merged (lo, hi) *inclusive* ranges in hphigh space (level 22).
    """
    ipix = np.sort(np.unique(np.asarray(indices, dtype=np.int64)))
    if len(ipix) == 0:
        return []
    shift = 2 * (HP_HIGH_LEVEL - level)
    los = ipix << shift
    his = ((ipix + 1) << shift) - 1
    ranges: list[tuple[int, int]] = []
    cur_lo, cur_hi = int(los[0]), int(his[0])
    for lo, hi in zip(los[1:], his[1:]):
        lo, hi = int(lo), int(hi)
        if lo <= cur_hi + 1:
            cur_hi = max(cur_hi, hi)
        else:
            ranges.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    ranges.append((cur_lo, cur_hi))
    return ranges


def get_image_filepaths() -> list[str]:
    if IMAGE_DB_PATH.exists():
        query = duckdb.sql(f"SELECT filepath FROM {DUCK_IMAGE}")
        output = query.fetchnumpy()["filepath"]
        return [str(x) for x in output]
    else:
        return []

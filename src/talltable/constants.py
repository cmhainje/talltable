import numpy as np

PART_MIN_LEVEL = 6
PART_MAX_LEVEL = 10
HP_HIGH_LEVEL = 22

_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing="ij"))

MAX_ROWS_PER_PART = 200_000_000


# HP_PART_LEVEL = 6
# DF_PART_LEVEL = 9
# DF_SAFE_RADIUS = 6
# 
# from astropy.coordinates import SkyCoord
# NORTH_DEEP_FIELD = SkyCoord(lat="90d", lon="0d", frame="geocentricmeanecliptic")
# SOUTH_DEEP_FIELD = SkyCoord(lat="-82d", lon="+44.8d", frame="geocentricmeanecliptic")

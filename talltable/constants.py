import numpy as np
from astropy.coordinates import SkyCoord

HP_PART_LEVEL = 6
DF_PART_LEVEL = 9
HP_HIGH_LEVEL = 22
DF_SAFE_RADIUS = 6

_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing="ij"))

NORTH_DEEP_FIELD = SkyCoord(lat="90d", lon="0d", frame="geocentricmeanecliptic")
SOUTH_DEEP_FIELD = SkyCoord(lat="-82d", lon="-44.8d", frame="geocentricmeanecliptic")

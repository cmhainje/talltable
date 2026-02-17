import numpy as np
from astropy.coordinates import SkyCoord

_idx = np.arange(2040, dtype=np.uint32)
ALL_ROW, ALL_COL = map(np.ravel, np.meshgrid(_idx, _idx, indexing="ij"))
HP_PART_C_LEVEL = 4
HP_PART_F_LEVEL = 7
HP_HI_LEVEL = 22
N_WAVEPART = 24

NORTH_DEEP_FIELD = SkyCoord(lat="90d", lon="0d", frame="geocentricmeanecliptic")
SOUTH_DEEP_FIELD = SkyCoord(lat="-82d", lon="-44.8d", frame="geocentricmeanecliptic")

import numpy as np
import healpy as hp
from .constants import (
    DF_PART_LEVEL,
    HP_PART_LEVEL,
    DF_SAFE_RADIUS,
    NORTH_DEEP_FIELD,
    SOUTH_DEEP_FIELD,
)


def to_uvec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=1)


_north_icrs = NORTH_DEEP_FIELD.transform_to("icrs")
_south_icrs = SOUTH_DEEP_FIELD.transform_to("icrs")
DEEP_FIELDS = to_uvec(
    np.array([_north_icrs.ra.deg, _south_icrs.ra.deg]),
    np.array([_north_icrs.dec.deg, _south_icrs.dec.deg]),
)
DEEP_FIELD_COS_RADIUS = np.cos(np.radians(DF_SAFE_RADIUS))  # cos(6 deg)


def partition(ra, dec):
    scalar_input = np.ndim(ra) == 0
    ra, dec = map(np.atleast_1d, (ra, dec))

    df_part = hp.ang2pix(2**DF_PART_LEVEL, ra, dec, nest=True, lonlat=True)
    hp_part = df_part >> (2 * (DF_PART_LEVEL - HP_PART_LEVEL))

    # if not in deep field, set dfpart to a bad value
    in_df = np.any((DEEP_FIELDS @ to_uvec(ra, dec).T) > DEEP_FIELD_COS_RADIUS, axis=0)
    df_part[~in_df] = -1

    if scalar_input:
        return hp_part.squeeze(), df_part.squeeze()
    return hp_part, df_part

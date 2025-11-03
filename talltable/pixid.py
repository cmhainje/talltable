import numpy as np

BITMASK_COL = 2**11 - 2**0
BITMASK_ROW = 2**23 - 2**12
BITMASK_DET = 2**27 - 2**24


def rowcoldet_to_pixid(row, col, det):
    """given row, column, and detector, compute the pixid"""
    def numpyize(x):
        return np.atleast_1d(x).astype(np.int32)

    row, col, det = map(numpyize, (row, col, det))
    return (col + (row << 12) + (det << 24)).squeeze()


def pixid_to_rowcoldet(pixid):
    """given pixid, compute the row, column, and detector"""
    return (
        (pixid & BITMASK_ROW) >> 12,
        (pixid & BITMASK_COL),
        (pixid & BITMASK_DET) >> 24,
    )

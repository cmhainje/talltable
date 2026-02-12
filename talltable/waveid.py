import numpy as np

BITMASK_COL = 2**11 - 2**0
BITMASK_ROW = 2**23 - 2**12
BITMASK_DET = 2**27 - 2**24


def rowcoldet_to_waveid(row, col, det):
    """given row, column, and detector, compute the waveid"""
    def numpyize(x):
        return np.atleast_1d(x).astype(np.int32)

    row, col, det = map(numpyize, (row, col, det))
    return (col + (row << 12) + (det << 24)).squeeze()


def waveid_to_rowcoldet(waveid):
    """given waveid, compute the row, column, and detector"""
    return (
        (waveid & BITMASK_ROW) >> 12,
        (waveid & BITMASK_COL),
        (waveid & BITMASK_DET) >> 24,
    )

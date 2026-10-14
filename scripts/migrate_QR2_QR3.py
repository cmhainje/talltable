import argparse
from talltable.paths import PART_DB_PATH

QR2_DIR_POPEYE = "/mnt/sdceph/users/spherex/spherex_data_qr2"
QR3_DIR_POPEYE = "/mnt/sdceph/users/spherex/spherex_data_qr3"
QR2_DIR_LOCAL = "/Users/connor/research/talltable/data/qr2"
QR3_DIR_LOCAL = "/Users/connor/research/talltable/data/qr3"

FLUX_CORR = "{qr3}/l3_flux_corrections/cal-flxc-v1-2026-191/"
GAIN_MATRIX = "{qr3}/abs_gain_matrix/cal-agm-v8-2026-191/"
DARK_R6 = "{qr2}/dark/cal-drk-v4-2025-161/"
DARK_R7 = "{qr3}/dark/cal-drk-v6-2026-191/"
NONFUNC_MAP = "{qr3}/nonfunc_v2/base-2026-191/"
ZODI_SCALING = "{qr3}/zodi_scaling_v2/cal-zodi-v2-2026-254/"
SPECTRAL_WCS = "{qr3}/spectral_wcs/cal-swcs-v5-2026-191/"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    return ap.parse_args()


def download_if_missing(filepath):
    """checks if the data product is present. if not, downloads from S3."""
    raise NotImplementedError


def massage_data():
    """read the data products and produce arrays that are easier to use

    in particular, given the data products, compute per distinct waveid
    - c:     FLUX_CORR
    - delta: AGM_v8 * (DARK_v4 - DARK_v6)
    - zodi_scale: zodi_scaling_v2, interpolated/extrapolated
        interpolation should be linear in wavelength space per-detector
        extrapolation should clamp to the table (use nearest value)
        see notebooks/popeye/zodi_scaling_v2_validation.ipynb for evidence.
    - nonfunc: nonfunctional pixel map (boolean array)

    these should be shaped so that they can be indexed into given waveids later
    (note: I do not mean that they need to be seriously large arrays who can directly take a waveid as an index. just that, given a waveid, there should be a simple mapping to an index in this array. perhaps the index of the waveid if all distinct waveids were sorted and flattened? something.)

    note that nonfunc calibration product is in detector coordinates,
    so D4-6 are flipped along X relative to the sky.

    return c, delta, zodi_scale, nonfunc
    """

    # read data products
    # assert c is all finite and positive
    # interpolate/extrapolate zodi_scale to every wavelength

    raise NotImplementedError


def write_upgrade_table(c, delta, zodi_scale):
    """makes a new table in DB_DIR called qr2_to_qr3.parquet
    schema should be:
        waveid
        c: FLUX_CORR
        delta: AGM_v8 * (DARK_v4 - DARK_v6)
        zodi_scale: zodi_scaling_v2_D{n} inter-/extrapolated onto this waveid
    """
    raise NotImplementedError


def write_marker(part_number):
    """makes a file to the partition directory marking the partition as migrated from qr2 to qr3"""
    raise NotImplementedError


def has_been_migrated(part_number) -> bool:
    """checks for the existence of the 'migrated' marker"""
    raise NotImplementedError


def migrate_partition(part_number, c, delta, zodi_scale, nonfunc):
    """migrates partition `part_number`."""
    if has_been_migrated(part_number):
        return

    # read waveids and determine index arrays to index
    # into c, delta, etc for next parts

    # open new parquet file: `/path/to/part=xxx/migrating.parquet`

    # copy the data over, updating the columns that need updates
    # - be sure that the output remains sorted by hphigh (as the input was)
    # - also please use a row group size of 256k (currently ~1M)
    # write...
    #   flux     = old_flux     * c[indexes] + delta[indexes]
    #   variance = old_variance * c[indexes] * c[indexes]
    #   flags    = old_flags    | (NONFUNC_FLAG where nonfunc[indexes])
    #   zodi     = old_zodi     * zodi_scale[indexes]
    #   other cols = old_other_columns

    # write metadata to the footer:
    #   (fill in the metadata here)

    # replace migrating.parquet -> compacted.parquet

    write_marker(part_number)


def main():
    args = parse()
    paths = {
        "qr2": QR2_DIR_LOCAL if args.local else QR2_DIR_POPEYE,
        "qr3": QR3_DIR_LOCAL if args.local else QR3_DIR_POPEYE,
    }

    # download files
    download_if_missing(FLUX_CORR.format(**paths))
    download_if_missing(GAIN_MATRIX.format(**paths))
    download_if_missing(DARK_R6.format(**paths))
    download_if_missing(DARK_R7.format(**paths))
    download_if_missing(NONFUNC_MAP.format(**paths))
    download_if_missing(ZODI_SCALING.format(**paths))
    download_if_missing(SPECTRAL_WCS.format(**paths))

    # prepare upgrade data
    c, delta, zodi_scale, nonfunc = massage_data()
    write_upgrade_table(c, delta, zodi_scale)

    # load partitions
    with open(PART_DB_PATH, "r") as f:
        partitions = [int(x) for x in f.readlines()]

    # do the migrations
    for part_num in partitions:
        migrate_partition(part_num, c, delta, zodi_scale, nonfunc)


if __name__ == "__main__":
    main()

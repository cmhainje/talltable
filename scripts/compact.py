import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import h5py
import os

from os import remove
from glob import glob
from pathlib import Path
from tqdm import tqdm

from talltable.paths import PIXEL_DB_PATH


def main():
    index  = int(os.environ.get("SLURM_PROCID", 0))
    number = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"processing index {index} of {number} tasks")

    def part_num(path):
        return int(path.split('hppart=')[1])

    partitions = [Path(p) for p in sorted(
        glob(
            str(PIXEL_DB_PATH / "hppart=*")
        ), key=part_num
    )]

    # process only every Nth partition, starting on i
    if number > 1:
        partitions = partitions[index::number]

    for part in tqdm(partitions):
        def _h5_to_table(filepath):
            data = dict()
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:].squeeze()
            try:
                return pa.table(data)
            except pa.lib.ArrowInvalid as e:
                msg  = f"failed to processing h5 file {filepath}.\n"
                msg += "data dict included:\n"
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        msg += f"{key}: {data[key].shape}\n"
                    else:
                        msg += f"{key}: {data[key]}\n"
                msg += f"error message:\n{e}"
                raise RuntimeError(msg)

        try:
            # read in the HDF5 file(s), smoosh all together
            h5_files = glob(str(part / "chunk_*.hdf5"))
            if len(h5_files) == 0:
                continue
            table = pa.concat_tables([_h5_to_table(f) for f in h5_files])

            # sort
            sort_keys = [('hphigh', 'ascending')]
            table.sort_by(sort_keys)
            sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)

            # write out as parquet
            pq.write_table(
                table,
                part / 'compacted.parquet',
                compression='zstd',
                compression_level=3,
                sorting_columns=sorting_cols,
            )

            # clean up the intermediate HDF5 files
            for f in h5_files:
                remove(f)

        except RuntimeError as e:
            print(f"warning: failed processing partition {part}:\n{e}\ncontinuing...")


if __name__ == '__main__':
    main()

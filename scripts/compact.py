import pyarrow as pa
import pyarrow.parquet as pq
import h5py

from glob import glob
from pathlib import Path
from tqdm import tqdm

from talltable.paths import PIXEL_DB_PATH


def main():
    def part_num(path):
        return int(path.split('hppart=')[1])

    partitions = [Path(p) for p in sorted(
        glob(
            str(PIXEL_DB_PATH / "hppart=*")
        ), key=part_num
    )]

    for part in tqdm(partitions):

        # read in the HDF5 file(s)
        # smoosh all together, sort, write out as parquet

        def _h5_to_table(filepath):
            data = dict()
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:].squeeze()
            return pa.table(data)


        h5_files = glob(str(part / "chunk_*.hdf5"))
        table = pa.concat_tables([_h5_to_table(f) for f in h5_files])
        sort_keys = [('hphigh', 'ascending')]
        table.sort_by(sort_keys)
        sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)

        pq.write_table(
            table,
            part / 'compacted.parquet',
            compression='zstd',
            compression_level=3,
            sorting_columns=sorting_cols,
        )


if __name__ == '__main__':
    main()

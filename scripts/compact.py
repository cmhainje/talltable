import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import h5py
import os

from os import remove
from glob import glob
from pathlib import Path
from tqdm import tqdm

from talltable.paths import PIXEL_DB_PATH, IMAGE_DB_PATH, IMAGE_PARTS_DIR


def merge_image_parts():
    """Merge per-task image parquet files into the final IMAGE_DB_PATH."""
    part_files = sorted(glob(str(IMAGE_PARTS_DIR / "image_task*.parquet")))
    if not part_files:
        print("no image part files to merge")
        return

    print(f"merging {len(part_files)} image part files")
    tables = []
    if IMAGE_DB_PATH.exists():
        tables.append(pq.read_table(IMAGE_DB_PATH))
    for f in part_files:
        tables.append(pq.read_table(f))

    merged = pa.concat_tables(tables)
    tmp_file = Path(str(IMAGE_DB_PATH) + ".tmp")
    pq.write_table(merged, tmp_file)
    tmp_file.replace(IMAGE_DB_PATH)

    for f in part_files:
        remove(f)
    print(f"merged into {IMAGE_DB_PATH} ({len(merged)} rows)")


def main():
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"processing index {task_id} of {num_tasks} tasks")

    # merge image parts (only do once)
    if task_id == 0:
        merge_image_parts()

    partitions = [
        Path(p)
        for p in sorted(
            glob(str(PIXEL_DB_PATH / "hppart=*/dfpart=*/wavepart=*")),
        )
    ]

    # process only every Nth partition, starting on i
    if num_tasks > 1:
        partitions = partitions[task_id::num_tasks]

    for part in tqdm(partitions):

        def _h5_to_table(filepath):
            data = dict()
            with h5py.File(filepath, "r") as f:
                for key in f.keys():
                    data[key] = np.atleast_1d(f[key][:].squeeze())
            try:
                return pa.table(data)
            except pa.lib.ArrowInvalid as e:
                msg = f"failed to processing h5 file {filepath}.\n"
                msg += "data dict included:\n"
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        msg += f"{key}: np.array [{data[key].shape}]\n"
                    else:
                        msg += f"{key}: {data[key]}\n"
                msg += f"error message:\n{e}"
                raise RuntimeError(msg)

        try:
            # read in the HDF5 file(s), smoosh all together
            h5_files = glob(str(part / "chunk_*.hdf5"))
            if len(h5_files) == 0:
                continue

            pq_path = part / "compacted.parquet"

            tables = [_h5_to_table(f) for f in h5_files]
            if pq_path.exists():
                tables.append(pq.read_table(pq_path))
            table = pa.concat_tables(tables)

            # sort
            sort_keys = [("hphigh", "ascending")]
            table = table.sort_by(sort_keys)
            sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)

            # write out as parquet
            pq.write_table(
                table,
                pq_path,
                compression="zstd",
                compression_level=3,
                sorting_columns=sorting_cols,
            )

            # clean up the intermediate HDF5 files
            for f in h5_files:
                remove(f)

        except RuntimeError as e:
            print(f"warning: failed processing partition {part}:\n{e}\ncontinuing...")


if __name__ == "__main__":
    main()

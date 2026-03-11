import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np
import h5py
import os

from pathlib import Path
from tqdm import tqdm

from talltable.constants import HP_HIGH_LEVEL, MAX_ROWS_PER_PART, PART_MAX_LEVEL
from talltable.partition import part_to_level_index, level_index_to_part
from talltable.paths import PIXEL_DB_PATH, IMAGE_DB_PATH, IMAGE_PARTS_DIR


def merge_image_parts():
    """merge new partial image parquet files into the final images table"""
    part_files = sorted(IMAGE_PARTS_DIR.glob("image_task*.parquet"))
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
        f.unlink()
    print(f"merged into {IMAGE_DB_PATH} ({len(merged)} rows)")


def main():
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"processing index {task_id} of {num_tasks} tasks")

    # merge image parts (only do once)
    if task_id == 0:
        merge_image_parts()

    partitions = {}
    for p in sorted(PIXEL_DB_PATH.glob("part=*/*.hdf5")):
        part = int(p.parts[-2].split("=")[1])
        if part in partitions:
            partitions[part].append(p)
        else:
            partitions[part] = [p]

    # process only every Nth partition, starting on i
    keys = list(partitions.keys())
    if num_tasks > 1:
        keys = keys[task_id::num_tasks]

    for part in tqdm(keys):

        def _h5_to_table(filepath):
            data = dict()
            with h5py.File(filepath, "r") as f:
                for key in f.keys():
                    data[key] = np.atleast_1d(f[key][:].squeeze())
            try:
                return pa.table(data)
            except pa.lib.ArrowInvalid as e:
                msg = f"failed to process h5 file {filepath}.\n"
                msg += "data dict included:\n"
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        msg += f"{key}: np.array [{data[key].shape}]\n"
                    else:
                        msg += f"{key}: {data[key]}\n"
                msg += f"error message:\n{e}"
                raise RuntimeError(msg)

        try:
            level, index = part_to_level_index(part)

            # read in the HDF5 file(s), smoosh all together
            h5_files = partitions[part]
            if len(h5_files) == 0:
                continue

            pq_path = h5_files[0].with_name("compacted.parquet")
            tmp_path = pq_path.with_name("tmp.parquet")

            tables = [_h5_to_table(f) for f in h5_files]
            if pq_path.exists():
                try:
                    tables.append(pq.ParquetFile(pq_path).read())
                except pa.lib.ArrowInvalid as e:
                    msg = f"failed to open Parquet file {pq_path} with error message:\n{e}"
                    raise RuntimeError(msg)
            table = pa.concat_tables(tables)

            # sort
            sort_keys = [("hphigh", "ascending")]
            table = table.sort_by(sort_keys)
            sorting_cols = pq.SortingColumn.from_ordering(table.schema, sort_keys)


            # check if its too big
            if len(table) > MAX_ROWS_PER_PART and level < PART_MAX_LEVEL:
                # split into 4 subpartitions at the next level
                _level = level + 1
                for k in range(4):
                    _index = (index << 2) + k
                    _p = level_index_to_part(_level, _index)
                    _part_dir = PIXEL_DB_PATH / f"part={_p}"
                    _part_dir.mkdir(exist_ok=False)

                    _lo = (_index)     << 2 * (HP_HIGH_LEVEL - _level)
                    _hi = (_index + 1) << 2 * (HP_HIGH_LEVEL - _level)

                    _mask = (pc.field("hphigh") >= _lo) & (pc.field("hphigh") < _hi)
                    _table = table.filter(_mask)

                    _pq_path = _part_dir / "compacted.parquet"
                    pq.write_table(
                        _table,
                        _pq_path,
                        compression="zstd",
                        compression_level=3,
                        sorting_columns=sorting_cols,
                    )

                # @TODO: store the full list of partitions somewhere, update it here

                # clean up
                for f in h5_files:
                    f.unlink()
                pq_path.unlink()
                pq_path.parent.rmdir()



            else:
                pq.write_table(
                    table,
                    tmp_path,
                    compression="zstd",
                    compression_level=3,
                    sorting_columns=sorting_cols,
                )

                # clean up
                for f in h5_files:
                    f.unlink()
                tmp_path.replace(pq_path)


        except RuntimeError as e:
            print(f"warning: failed processing partition {part}:\n{e}\ncontinuing...")


if __name__ == "__main__":
    main()

---
title: SPHEREx talltable documentation
---

# Authors

This database and associated package are authored by

- **Connor Hainje**, NYU
- **David W Hogg**, NYU+

Contact email: `connor <dot> hainje <at> nyu <dot> edu`


# Installation

The project can be installed by

```
git clone https://github.com/cmhainje/talltable.git
```

Inside, there is a Python package, `talltable`, which handles building and querying the database.
Install it from inside the project with

```
uv sync
```

To test that it installed correctly,

```
$ uv run python -c "import talltable; print(talltable)"
<module 'talltable' from '/path/to/talltable/src/talltable/__init__.py'>
```


# Schema

## Waves

The **waves** table stores the wavelength information.
Its filename is `waves.parquet`.
The data comes from the most recent `spectral_wcs` data released by SPHEREx.
The table schema is

```
waves: {
    waveid:     int
    wavelength: float
    bandwidth:  float
}
```

`waveid` is an integer that bit-packs the detector number (from 1 to 6) and the row and column (from 0 to 2039) of a pixel in its source image.
We provide utilities in the `talltable` package for translating between `waveid` and detector, row, column.

```python
>>> from talltable.waveid import rowcoldet_to_waveid, waveid_to_rowcoldet
>>> rowcoldet_to_waveid(479, 1831, 3)
52295463
>>> waveid_to_rowcoldet(52295463)
(479, 1831, 3)
```


## Images

The **images** table stores image metadata.
Its filename is `images.parquet`.
The table schema is 

```
images: {
    imageid:  int
    filepath: str
    @TODO
}
```

## Pixels

The meat of the database --- the namesake "tall table" --- is the **pixels** table.
This database is split over many files with filepaths like `pixels/part=${part}/compacted.parquet`.
The table schema is, roughly

```
pixels: {
    hphigh:   int64
    flux:     float32
    zodi:     float32
    variance: float32
    flags:    int32
    waveid:   int32
    imageid:  int64
    part:     int32
}
```

Details about each column are as follows:

- `hphigh` : a very level 22 HEALPix index giving the position on sky of the pixel.
- `flux`: the value of this pixel in the `"IMAGE"` layer of the spectral image file. Units: MJy/sr.
- `zodi`: the value of this pixel in the `"ZODI"` layer of the spectral image file. Units: MJy/sr.
- `variance`: the value of this pixel in the `"VARIANCE"` layer of the spectral image file. Units: (MJy/sr)$^2$.
- `flags`: the value of this pixel in the `"FLAGS"` layer of the spectral image file.
- `waveid`: a foreign key referencing the [waves table](/#waves). It encodes the detector, row, and column of this pixel.
- `imageid`: a foreign key referencing the [images table](/#images). It is the exposure ID number (`EXPIDN`) from the spectral image header.
- `part`: the partition number, encoding a much coarser HEALPix index for this pixel's location on sky. Note that this column is encoded in the _filepath_ of the partition, so the Parquet file itself will not contain a column `part`. More details [below](/#partitions).


## Partitions

For our partitioning scheme, we use HEALPix indices.
These start at level 6 ($Nside = 2^6$).
We set the maximum number of rows per partition to 200 million.
If a partition tries to exceed this number of rows, we subdivide it to the next HEALPix level, up to level 10.

The partition number packs both the HEALPix level and the pixel index into one 32-bit unsigned integer.
Note that HEALPix level $n$ has $12 \times 4^n$ pixels, therefore requiring $2 \, (n + 2)$ bits.
In the overhead, then, we can set a bit to encode the level.
We choose to set bit $2 \, (n + 4) + 1$: that's bit 29 for level 10, bit 21 for level 6.

In the nested ordering scheme, HEALPix indices can be downgraded by bitshifting right twice.
Notice how our level-encoding bit also shifts right twice when downgraded.
Thus, this property applies to our partition numbers as well!

```
level 10: 0b 0001 0000 xxxx xxxx xxxx xxxx xxxx xxxx
level 9:  0b 0000 0100 00xx xxxx xxxx xxxx xxxx xxxx
level 8:  0b 0000 0001 0000 xxxx xxxx xxxx xxxx xxxx
level 7:  0b 0000 0000 0100 00xx xxxx xxxx xxxx xxxx
level 6:  0b 0000 0000 0001 0000 xxxx xxxx xxxx xxxx
```

In the `talltable` package, we provide simple utitlity functions for translating between `part` and `(level, index)`.

```python
>>> from talltable.partition import level_index_to_part, part_to_level_index
>>> level_index_to_part(6, 12345)
1060921
>>> part_to_level_index(1060921)
(6, 12345)
```

We also provide a utility for finding the partition in the database which contains a point on sky (expects RA/Dec in degrees).

```python
>>> from talltable.partition import find_partition
>>> find_partition(350, 5)
@TODO: output
```

For larger queries, sometimes it can be useful to know all the partitions that have been created and at what levels.
We store this information in `partitions.txt`, which contains all the partition numbers separated by newlines.


# Building the database

Building the database proceeds in a few stages.

**Note:**
If you have access to the Popeye cluster, you can use my built version!
The files are located at `/mnt/sdceph/users/chainje/spxdb`.

The stages are

0. Configure paths to the source data and the database
1. Build the `waves` table
2. Ingest a batch
3. Compactify
4. Post-compaction clean-up
5. GOTO 2


## Configure paths

Do you have all the SPHEREx data on disk somewhere locally?
If so, set the path in `talltable/paths.py`:

```python
DATA_DIR = Path('/path/to/level2/')
```

Otherwise, we recommend streaming the data from S3.

```python
@TODO
```

Also, choose a location for the database to live.

```python
DB_DIR = Path('/path/to/spxdb')
```


## Build the waves table

The `waves` table can be built very simply:

```
uv run python scripts/build_wave.py
```


## Ingest

To ingest spectral images into the database, we first need to identify a batch of files.
Typically, we ingest images in a batch of one week of released data, in which case the list of files can be found with something like

```bash
find /path/to/level2/ -mindepth 4 -maxdepth 4 -name "*2025W19*" > batch.txt
```

Or, if you're using S3,

```
@TODO
```


Now, because Parquet (our final file format) is column-oriented, adding new rows is hard.
As such, ingestion proceeds in two stages.
First, we loop over all the spectral images, doing our processing and sorting the pixels into the relevant partitions.
This happens in the script `ingest.py`.

```
uv run python scripts/ingest.py batch.txt
```

This produces a bunch of small HDF5 files in the `pixels` directory.

Next, we loop over all the chunks, grouping together pixels from the same partition and writing them altogether into one Parquet file.
This happens in the script `compact.py`.

```
uv run python scripts/compact.py
```

We also have a script that builds a file, `parts.txt`, with all the known partition numbers in it.
It should be run after every compaction, as it is needed for the ingestion step.

```
uv run python scripts/build_parts.py
```

This script also cleans up the leftover HDF5 files.


Note that these scripts are set up to parallelize work over SLURM tasks.
It is as easy as using an SLURM submission script like

```
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-tasks 1

srun uv run python scripts/ingest.py batch.txt
srun uv run python scripts/compact.py
```

Note: don't run `build_parts.py` or `build_wave.py` with `srun`, because the different tasks will fight each other.
See our [one-week SLURM script](https://github.com/cmhainje/talltable/blob/main/slurm/week.sbatch) for an example that goes end-to-end.



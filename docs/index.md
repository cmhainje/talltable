---
title: NYU SPHEREx talltable documentation
---

# Authors

This database and associated package are authored by

- **Connor Hainje**, NYU
- **David W Hogg**, NYU+

Contact email: `connor <dot> hainje <at> nyu <dot> edu`


# Getting started

## TL;DR

```
# install
git clone https://github.com/cmhainje/talltable.git
cd talltable
uv sync

# configure
cp example_config.toml config.toml
# <EDIT CONFIG>

# pull down (part of) the database using globus
uv run python scripts/globus.py disc "06h33m45s" "04d59m54s" 3.0
# <REQUIRES A GLOBUS ACCOUNT & PERMISSION, EMAIL ME>
```

## Installation

The project can be installed by

```
git clone https://github.com/cmhainje/talltable.git
cd talltable
```

Inside, there is a Python package, `talltable`, which handles building and interfacing with the database.
Install the package and all dependencies from inside the project with

```
uv sync
```

To test that it installed correctly,

```
uv run python -c "import talltable; print('ok!')"
ok!
```


## Configuration

Next, you need to make a `config.toml` directory in the project root.
There's an example; copy it:

```
cp example_config.toml config.toml
```

and then replace the paths with your own.

- `data_dir` is only needed if you're going to download raw SPHEREx FITS files and use it to build the database yourself. (Set to a dummy path if you don't need it.)
- `db_dir` is where the database files live.
- `scratch_dir` is where transient files produced during ingestion live. (Unless you have reason to do something else, make this a dummy path or set it to the same path as `db_dir`.)


# Access

If you only want to *use* the database (and not build it yourself), there are several ways to do so.

## Popeye cluster

If you are a researcher at the Flatiron Institute with access to the Popeye cluster, the files are all available locally!
Set the following in your `config.toml`:

```toml
[paths]
data_dir    = "/mnt/sdceph/users/spherex/spherex_data_qr2"
db_dir      = "/mnt/sdceph/users/spherex/talltable"
scratch_dir = "/mnt/sdceph/users/spherex/talltable"
```

## Globus

Our pre-built version of the database is also available on Globus.
You will need permissions in Globus: email me for access.

You don't need to download all of the database; to execute queries, you only need
`parts.txt`, `image.parquet`, `waves.parquet`,
and whichever partitions in `pixels` are relevant to your region on sky.

The script `scripts/globus.py` automates this download.
You define a sky region as a disc or rectangle (centered at some RA/Dec with some radius or width/height), or give a list of HEALPix indices.
The script then downloads only the necessary partitions, putting them into the `db_dir` specified in your config.
This script is good for relatively small downloads; let me know if you're trying to pull down a large fraction of the data, and we can find a better solution.


## Web service

We will (soon) release a public web service with an endpoint to execute SQL queries against the database.
Note that this *will* be limited with relatively stringent timeouts.
If you are doing heavy analysis in your queries, or if you are making a large number of queries on one region, you will be better served by pulling down the chunks of the database you need from Globus.



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
We choose to set bit $2 \, (n + 4)$: that's bit 28 for level 10, bit 21 for level 6.

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
We store this information in `parts.txt`, which contains all the partition numbers separated by newlines.


# Queries

## Query builder

We will (soon) release a query builder which constructs common kinds of queries with the fastest structure that we know of.

@TODO: add documentation


## Cookbook

The general approach to building a fast query is the following:

1. Identify and load only the relevant partitions
2. Load the wavelength table into memory ahead-of-time
3. Join against only the part of the wavelength table you need (rather than joining against the whole table and then using a `WHERE` condition)
4. Use `GROUP BY` to aggregate data over HEALPixels (if relevant to your query)

@TODO: add common queries here


# Building the database yourself

Building the database proceeds in a few stages.

The stages are

1. Build the `waves` table
2. Ingest a batch
3. Compactify
4. Post-compaction clean-up
5. GOTO 2

Make sure you have set up your `config.toml` ([see here](/#configuration)) before you start!

First, build the waves table. This is done with:

```
uv run python scripts/build_waves.py
```

Then, ingest some images.
To ingest spectral images, we first need to identify a batch of files.
Typically, we ingest images in a batch of one week of released data, in which case the list of files can be found with something like

```bash
find /path/to/level2/ -mindepth 4 -maxdepth 4 -name "*2025W19*" > batch.txt
```

Alternatively, make `batch.txt` with all files that intersect some point on sky, if you only want to ingest the data in a given region.
(@TODO: add TAP query)

Now, because Parquet (our final file format) is column-oriented, adding new rows is hard.
As such, ingestion proceeds in two stages.
First, we loop over all the spectral images, doing our processing and sorting the pixels into the relevant partitions.
This happens in the script `ingest.py`.

```
uv run python scripts/ingest.py batch.txt
```

This produces a bunch of small binary files in the `pixels` directory.

Next, we loop over all the chunks, grouping together pixels from the same partition and writing them all together into one Parquet file.
This happens in the script `compact.py`.

```
uv run python scripts/compact.py
```

The ingestion and compaction steps produce many transient files that should be deleted after the steps are complete.
The post-compaction script cleans this up, as well as builds a file `parts.txt` with the known the partition numbers in it.

```
uv run python scripts/post_compact.py
```

Note that these scripts are set up to parallelize work over SLURM tasks.
It is as easy as using an SLURM submission script like

```
#SBATCH --nodes <N>
#SBATCH --ntasks-per-node <M>
#SBATCH --cpus-per-tasks 1

srun uv run python scripts/ingest.py batch.txt
srun uv run python scripts/compact.py
```

This will break up the ingestion and compaction work over `N * M` tasks.

Note: don't run `build_waves.py` or `post_compact.py` with `srun`, because the different tasks will fight each other.
See our [one-week SLURM script](https://github.com/cmhainje/talltable/blob/main/slurm/week.sbatch) for an example that goes end-to-end.



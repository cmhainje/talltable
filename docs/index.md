---
title: NYU SPHEREx talltable documentation
---

# Authors

This database and associated package(s) are authored by

- **Connor Hainje**, NYU
- **David W Hogg**, NYU+

Contact email: `connor <dot> hainje <at> nyu <dot> edu`


# Quick start

There are three main ways to interact with the talltable.

1. Querying the database via the web service.
2. Querying the database locally. This requires either
    access to the Flatiron Institute cluster where the full database lives, or 
    for you to download the chunks of the database that you need.
3. Building the database from the SPHEREx spectral image files yourself.

If you want to do option 3, read [Building the database yourself](/#building-the-database-yourself).

Otherwise, start by installing the core talltable package:

```
uv add 'talltable @ git+https://github.com/cmhainje/talltable.git#subdirectory=packages/core'
```

or, if you use pip:

```
pip install 'talltable @ git+https://github.com/cmhainje/talltable.git#subdirectory=packages/core'
```

This package provides a client for the web service as well as a number of utilities that are helpful for building queries.


## Using the web service

Out of the box, you should be able to use the `PixelQuery` class to build and execute simple queries.

```python
from talltable import PixelQuery

table = (
    PixelQuery(web=True)
    .disc(
        98.4,  # ra [deg]
        5.0,   # dec [deg]
        30.0   # radius [arcmin]
    )  
    .with_wavelengths()
    .execute()
)
```

More details on the `PixelQuery` class [below](/#query-builder).


## Local queries

In order to query *local* database files, you need to configure talltable so that it knows where the database lives.
This is done with the environment variable `TALLTABLE_DB_DIR`.
Set this however you like:

- Make a `.env` file and pass it to uv whenever you run commands (`uv run --env-file .env ...`)
- Set `export TALLTABLE_DB_DIR=/path/to/db` in your `.bashrc` or `.zshrc` (or whatever shell you used)
- Or however else you want to manage env vars for your project.

**For Flatiron users:** use `TALLTABLE_DB_DIR=/mnt/sdceph/users/spherex/talltable`.

Then, you can use the `PixelQuery` class to build and execute simple queries.

```python
from talltable import PixelQuery

table = (
    PixelQuery(web=False)
    .disc(
        98.4,  # ra [deg]
        5.0,   # dec [deg]
        30.0   # radius [arcmin]
    )  
    .with_wavelengths()
    .execute()
)
```


# Access

There are several ways to access the database, as hinted at above.

## Web service

There is a web service at `https://talltable.flatironinstitute.org`.
It has only a handful of endpoints, and requires use via the client provided in the `talltable` package.
The web service has fairly strict compute, memory, and time limits on queries.
If you want to run large queries, you may be better serviced by downloading parts of the pre-built database yourself!


## Downloading chunks for local use

The database is split into chunks based called 'partitions', which individually are Parquet files.
The scheme for deciding on these partitions is described [below](/#partitions);
briefly, the sky is partitioned by adaptively-refined HEALPix indices.
We provide a few methods to download partitions in their entirety.

1. Download the files from
    [https://sdsc-users.flatironinstitute.org/~chainje/talltable/](https://sdsc-users.flatironinstitute.org/~chainje/talltable/).
    These can be downloaded with curl or wget or whatever you like.
2. We are set up on Globus! This requires permissions, email me for access.

**Important:** if you are downloading a part of the database for local use, you need to

- Also download the files `parts.txt`, `image.parquet`, and `waves.parquet`
- Replicate the directory structure within the database directory exactly
- Specify via environment variable (`TALLTABLE_DB_DIR`) the path to your local database files

There is a script, `scripts/globus.py`, which handles everything for you.
You just need to specify the database directory environment variable and provide a region on sky;
the script computes which partitions to pull down and makes sure everything is in the right place.


## Popeye cluster

If you are a researcher at the Flatiron Institute with access to the Popeye cluster, the files are all available locally!
Set the following environment variable:

```
TALLTABLE_DB_DIR=/mnt/sdceph/users/spherex/talltable
```

You should be good to go.


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
>>> from talltable import rowcoldet_to_waveid, waveid_to_rowcoldet
>>> rowcoldet_to_waveid(479, 1831, 3)
52295463
>>> waveid_to_rowcoldet(52295463)
(479, 1831, 3)
```


## Images

The **images** table stores image metadata.
Its filename is `image.parquet`.
The table schema is 

```
images: {
    imageid:  int64
    filepath: str
    obsid:    str
    t_beg:    float64
    t_end:    float64
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

- `hphigh` : a level 22 HEALPix index giving the position on sky of the pixel.
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

For larger queries, sometimes it can be useful to know all the partitions that have been created and at what levels.
We store this information in `parts.txt`, which contains all the partition numbers separated by newlines.


# Queries

## Query builder

In the core `talltable` package, we provide a query builder called `PixelQuery`.
It can be used to query the web service or local files.
Here are all the features it supports:

```python
query = (
    PixelQuery(web=True)  # or web=False if using local data

    # set a region using one of these three (required)
    .disc(ra, dec, radius)         # cone search
    .rect(ra, dec, width, height)  # rect search
    .ipix(pixels, level)           # grab data by healpix indices

    # add filters
    .flags(mask_known_source=False)  # filter by flags
    .wavelength(wave_min, wave_max)  # filter to a wavelength range

    # add extra output columns
    .with_wavelengths()  # add wavelength and bandwidth columns
    .with_image()        # add image filepath column
    .with_rowcoldet()    # add row, column, detector columns
)

```

This produces a `PixelQuery` object. You can view the SQL it will generate

```python
print(query.sql())
```

You can directly execute the query.

```python
query.execute()  # -> returns a PyArrow Table
query.execute_to_parquet(output_filepath)  # saves outputs to a Parquet file
```




## Cookbook

The general approach to building a fast query is the following:

1. Identify and load only the relevant partitions
2. Load the wavelength table into memory ahead-of-time
3. Join against only the part of the wavelength table you need (rather than joining against the whole table and then using a `WHERE` condition)
4. Use `GROUP BY` to aggregate data over HEALPixels (if relevant to your query)

@TODO: add common queries here


# Building the database yourself

Building the database proceeds in a few stages.

First, make sure you have your environment variables set. You will need

- `TALLTABLE_DB_DIR`: tells the system where to build the database
- `TALLTABLE_DATA_DIR`: path to the spectral images
- `TALLTABLE_SCRATCH_DIR` (optional): put interim files in a different location.
    This is good if you have very fast scratch space available.
    Defaults to `TALLTABLE_DB_DIR`.

The stages are

1. Build the `waves` table
2. Ingest a batch
3. Compactify
4. Post-compaction clean-up
5. GOTO 2

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



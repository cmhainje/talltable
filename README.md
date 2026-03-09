# SPHEREx talltable

## authors

- **Connor Hainje**, NYU
- **David W Hogg**, NYU+

## installation

```bash
pip install git+https://github.com/cmhainje/talltable
```

or, for local dev,

```bash
git clone https://github.com/cmhainje/talltable.git
cd spxperiments
poetry install
```

## description

we are trying to build a database of the individual pixels in the SPHEREx images.
the desired schema is a SQL table with ~10 columns and one row per pixel.
because there will be ~10 trillion pixels in the SPHEREx data, this will be quite a tall table.

utilities worth using include...

- `scripts/download.py`
    - efficiently fetch and download all spectral images from a given MJD.
    - uses one query to the IRSA TAP service to determine the list of image URLs.
    - then, downloads those URLs either from IRSA (via GET requests) or the AWS data mirror (via S3).
    - the download loop can be parallelized with as many workers as you like.
    - sample usage: `python download.py 60907 -N 24 --use-aws`
- `scripts/ingest.py`
    - build an Arrow Dataset of Parquet files from the pixels in your downloaded images.
    - running a second time will ingest only the new ones.
      note that each run of `ingest.py` produces a new 'release' (eg, a new set of Parquet files).
      (we will soon provide a compaction script to combine releases.)
    - sample usage: `python ingest.py`


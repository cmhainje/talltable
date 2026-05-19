from __future__ import annotations

import healpy as hp
import numpy as np
import pyarrow as pa

from pathlib import Path

from talltable.constants import HP_HIGH_LEVEL, PART_MAX_LEVEL, SERVICE_URL
from talltable.partition import find_partitions_disc, find_partitions_rect, find_partitions_ipix
from talltable.paths import PART_DB_PATH, PIXEL_DB_PATH, WAVES_DB_PATH, IMAGE_DB_PATH


_KNOWN_SOURCE_BIT = 1 << 21  # bit 21: pixel is near a known source

_MAX_FILTER_RANGES = 500
_FILTER_TARGET_PIXELS = 1000


def _choose_filter_level(region) -> int:
    """Choose HEALPix level L so the region spans ~_FILTER_TARGET_PIXELS pixels."""
    kind = region[0]
    if kind == "disc":
        _, ra, dec, radius_deg = region
        area = np.pi * np.radians(radius_deg) ** 2
    elif kind == "rect":
        _, ra, dec, width_deg, height_deg = region
        area = np.radians(width_deg) * np.radians(height_deg)
    else:
        return HP_HIGH_LEVEL
    # N_pix ≈ 3 * area_rad² * 4^L  →  4^L = TARGET*π / (3*area)
    nside_sq = _FILTER_TARGET_PIXELS * np.pi / (3.0 * area)
    L = round(np.log2(nside_sq) / 2)
    return max(PART_MAX_LEVEL, min(HP_HIGH_LEVEL, L))


def _indices_to_hphigh_ranges(ipix_L: np.ndarray, shift: int) -> list[tuple[int, int]]:
    """Convert sorted level-L pixel indices to merged hphigh (lo, hi) inclusive ranges."""
    ipix_L = np.sort(np.unique(ipix_L)).astype(np.int64)
    if len(ipix_L) == 0:
        return []
    los = ipix_L << shift
    his = ((ipix_L + 1) << shift) - 1
    ranges: list[tuple[int, int]] = []
    cur_lo, cur_hi = int(los[0]), int(his[0])
    for lo, hi in zip(los[1:], his[1:]):
        lo, hi = int(lo), int(hi)
        if lo <= cur_hi + 1:
            cur_hi = max(cur_hi, hi)
        else:
            ranges.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    ranges.append((cur_lo, cur_hi))
    return ranges


def _ipix_hphigh_ranges(indices, level: int) -> list[tuple[int, int]]:
    """Convert user-supplied HEALPix indices at the given level to hphigh ranges."""
    shift = 2 * (HP_HIGH_LEVEL - level)
    return _indices_to_hphigh_ranges(np.asarray(indices, dtype=np.int64), shift)


def _disc_rect_hphigh_ranges(region) -> list[tuple[int, int]] | None:
    """
    Compute hphigh (lo, hi) inclusive ranges that cover the region.

    Chooses a filter HEALPix level L so the region spans ~1000 pixels, queries
    those pixels with inclusive=True, then converts each contiguous run of
    level-L indices to a single BETWEEN range in hphigh space (nested HEALPix
    guarantees index i at level L occupies exactly hphigh in
    [i<<shift, (i+1)<<shift - 1]).

    Returns None when the range count exceeds _MAX_FILTER_RANGES, meaning the
    region is large enough that partition selection is already a sufficient
    approximation and adding a filter would generate unwieldy SQL.
    """
    level = _choose_filter_level(region)
    nside = 2 ** level
    shift = 2 * (HP_HIGH_LEVEL - level)

    kind = region[0]
    if kind == "disc":
        _, ra, dec, radius_deg = region
        ipix_L = hp.query_disc(
            nside, hp.ang2vec(ra, dec, lonlat=True),
            np.radians(radius_deg), nest=True, inclusive=True,
        )
    else:  # rect
        _, ra, dec, width_deg, height_deg = region
        corners = hp.ang2vec(
            ra + 0.5 * width_deg * np.array([-1, +1, +1, -1]),
            dec + 0.5 * height_deg * np.array([-1, -1, +1, +1]),
            lonlat=True,
        )
        ipix_L = hp.query_polygon(nside, corners, nest=True, inclusive=True)

    ranges = _indices_to_hphigh_ranges(ipix_L, shift)
    return ranges if len(ranges) <= _MAX_FILTER_RANGES else None


class PixelQuery:
    def __init__(self, web: bool = False, base_url: str = SERVICE_URL):
        self._web = web
        self._base_url = base_url.rstrip("/")
        self._region = None
        self._wavelength = None
        self._flags_sql = f"(flags & ~({_KNOWN_SOURCE_BIT})) = 0"
        self._with_rowcoldet = False
        self._with_image = False
        self._with_radec = False

    # --- region ---

    def disc(self, ra: float, dec: float, radius: float) -> PixelQuery:
        """Circular region on sky. ra/dec in degrees, radius in arcmin."""
        self._region = ("disc", ra, dec, radius * 60.0)
        return self

    def rect(self, ra: float, dec: float, width: float, height: float) -> PixelQuery:
        """Rectangular region on sky. ra/dec is center, width/height in arcmin."""
        self._region = ("rect", ra, dec, width * 60.0, height * 60.0)
        return self

    def ipix(self, indices: list[int], level: int) -> PixelQuery:
        """Specific HEALPix pixels at the given nesting level."""
        self._region = ("ipix", indices, level)
        return self

    # --- filters ---

    def wavelength(self, min_um: float, max_um: float) -> PixelQuery:
        """Filter to pixels whose center wavelength falls within [min_um, max_um] microns."""
        self._wavelength = (min_um, max_um)
        return self

    def flags(
        self,
        *,
        mask_known_source: bool = False,
        custom_mask: int | None = None,
    ) -> PixelQuery:
        """
        Control flag filtering.

        Default (no args): accept pixels where no bits other than bit 21 are set,
            i.e. flags & ~(1<<21) == 0.
        mask_known_source=True: also require bit 21 to be clear, i.e. flags == 0.
        custom_mask=M: accept pixels where flags & M == 0 (overrides the above).
        """
        if mask_known_source and custom_mask is not None:
            raise ValueError("mask_known_source and custom_mask are mutually exclusive")
        if custom_mask is not None:
            self._flags_sql = f"(flags & {custom_mask}) = 0"
        elif mask_known_source:
            self._flags_sql = "flags = 0"
        else:
            self._flags_sql = f"(flags & ~({_KNOWN_SOURCE_BIT})) = 0"
        return self

    # --- extra columns ---

    def with_rowcoldet(self) -> PixelQuery:
        """Add col, row, det columns decoded from waveid."""
        self._with_rowcoldet = True
        return self

    def with_image(self) -> PixelQuery:
        """Join against the images table to include filepath for each pixel."""
        self._with_image = True
        return self

    def with_radec(self) -> PixelQuery:
        """Add ra, dec columns computed from hphigh (applied client-side after execute)."""
        self._with_radec = True
        return self

    # --- partitions ---

    def partitions(self) -> list[int]:
        """Return the list of partition indices covering the requested region."""
        if self._region is None:
            raise ValueError("Region must be specified before calling partitions()")
        if self._web:
            return self._web_partitions()
        else:
            return self._local_partitions()

    def _local_partitions(self) -> list[int]:
        with open(PART_DB_PATH) as f:
            all_parts = set(int(line.strip()) for line in f)
        kind = self._region[0]
        if kind == "disc":
            _, ra, dec, radius = self._region
            parts = find_partitions_disc(ra, dec, radius, all_parts=all_parts)
        elif kind == "rect":
            _, ra, dec, width, height = self._region
            parts = find_partitions_rect(ra, dec, width, height, all_parts=all_parts)
        elif kind == "ipix":
            _, indices, level = self._region
            parts = find_partitions_ipix(indices, level, all_parts=all_parts)
        else:
            raise ValueError(f"Unknown region type: {kind}")
        return sorted(map(int, parts))

    def _web_partitions(self) -> list[int]:
        import requests
        kind = self._region[0]
        if kind == "disc":
            _, ra, dec, radius = self._region
            resp = requests.get(
                f"{self._base_url}/partitions/disc",
                params={"ra": ra, "dec": dec, "radius": radius},
            )
        elif kind == "rect":
            _, ra, dec, width, height = self._region
            resp = requests.get(
                f"{self._base_url}/partitions/rect",
                params={"ra": ra, "dec": dec, "width": width, "height": height},
            )
        elif kind == "ipix":
            _, indices, level = self._region
            resp = requests.post(
                f"{self._base_url}/partitions",
                json={"level": level, "indices": indices},
            )
        else:
            raise ValueError(f"Unknown region type: {kind}")
        resp.raise_for_status()
        return sorted(resp.json()["partitions"])

    # --- SQL ---

    def sql(self) -> str:
        """Return the SQL query string."""
        if self._region is None:
            raise ValueError("Region must be specified before calling sql()")
        return self._build_sql("pixels", "waves", "images")

    def _local_sql(self, parts: list[int]) -> str:
        paths = [f"'{PIXEL_DB_PATH / f'part={p}/compacted.parquet'}'" for p in parts]
        if len(paths) == 1:
            pixel_source = f"read_parquet({paths[0]})"
        else:
            pixel_source = f"read_parquet([{', '.join(paths)}])"
        waves_source = f"read_parquet('{WAVES_DB_PATH}')"
        images_source = f"read_parquet('{IMAGE_DB_PATH}')"
        return self._build_sql(pixel_source, waves_source, images_source)

    def _build_sql(self, pixel_source: str, waves_source: str, images_source: str) -> str:
        # SELECT columns
        cols = [
            "p.waveid", "p.flux", "p.variance", "p.zodi",
            "p.flags", "p.hphigh", "p.imageid",
        ]
        if self._with_rowcoldet:
            cols += [
                "(p.waveid & 2047) AS col",
                "((p.waveid >> 12) & 2047) AS row",
                "((p.waveid >> 24) & 7) AS det",
            ]
        if self._with_image:
            cols.append("i.filepath")

        select = "SELECT\n    " + ",\n    ".join(cols)

        # CTE for wavelength pre-filter
        cte = None
        if self._wavelength is not None:
            min_um, max_um = self._wavelength
            cte = (
                f"WITH selected_waves AS (\n"
                f"    SELECT * FROM {waves_source}"
                f" WHERE wavelength BETWEEN {min_um} AND {max_um}\n"
                f")"
            )

        # FROM + JOINs
        froms = [f"FROM {pixel_source} p"]
        if self._wavelength is not None:
            froms.append("JOIN selected_waves w ON p.waveid = w.waveid")
        if self._with_image:
            froms.append(f"LEFT JOIN {images_source} i ON p.imageid = i.imageid")

        # WHERE conditions
        conditions = []
        kind = self._region[0]
        if kind == "ipix":
            _, indices, level = self._region
            ranges = _ipix_hphigh_ranges(indices, level)
            comment = f"-- ipix, level {level}"
            clauses = [f"(p.hphigh BETWEEN {lo} AND {hi})" for lo, hi in ranges]
            body = clauses[0] if len(clauses) == 1 else "(\n    " + "\n    OR ".join(clauses) + "\n  )"
            conditions.append(comment + "\n  " + body)
        elif kind in ("disc", "rect"):
            ranges = _disc_rect_hphigh_ranges(self._region)
            if ranges:
                filter_level = _choose_filter_level(self._region)
                if kind == "disc":
                    _, ra, dec, radius = self._region
                    comment = f"-- disc(ra={ra}, dec={dec}, radius={radius}°), level-{filter_level} hphigh ranges"
                else:
                    _, ra, dec, width, height = self._region
                    comment = f"-- rect(ra={ra}, dec={dec}, width={width}°, height={height}°), level-{filter_level} hphigh ranges"
                clauses = [f"(p.hphigh BETWEEN {lo} AND {hi})" for lo, hi in ranges]
                body = clauses[0] if len(clauses) == 1 else "(\n    " + "\n    OR ".join(clauses) + "\n  )"
                conditions.append(comment + "\n  " + body)

        conditions.append(f"-- quality flags\n  {self._flags_sql}")

        where_lines = ["WHERE"]
        for i, cond in enumerate(conditions):
            if cond.startswith("--"):
                comment, expr = cond.split("\n  ", 1)
                where_lines.append(f"  {comment}")
                where_lines.append(f"  AND {expr}" if i > 0 else f"  {expr}")
            else:
                where_lines.append(f"  AND {cond}" if i > 0 else f"  {cond}")
        where = "\n".join(where_lines)

        parts = []
        if cte is not None:
            parts.append(cte)
        parts += [select] + froms + [where]
        return "\n".join(parts)

    # --- execute ---

    def execute(self) -> pa.Table:
        """Run the query and return an Arrow Table."""
        parts = self.partitions()
        if self._web:
            from talltable.client import fetch_arrow_stream
            table = fetch_arrow_stream(
                f"{self._base_url}/sql",
                {"query": self.sql(), "partitions": parts},
            )
        else:
            import duckdb
            con = duckdb.connect()
            try:
                table = con.execute(self._local_sql(parts)).fetch_arrow_table()
            finally:
                con.close()

        if self._with_radec:
            table = _add_radec(table)
        return table

    def execute_to_parquet(self, output: str | Path) -> None:
        """Run the query and stream the result directly to a Parquet file."""
        if self._with_radec:
            raise ValueError(
                "with_radec() is not supported with execute_to_parquet(); "
                "use execute() instead and write the table manually."
            )
        parts = self.partitions()
        if self._web:
            from talltable.client import fetch_arrow_to_parquet
            fetch_arrow_to_parquet(
                f"{self._base_url}/sql",
                {"query": self.sql(), "partitions": parts},
                output,
            )
        else:
            import duckdb
            import pyarrow.parquet as pq
            con = duckdb.connect()
            try:
                reader = con.execute(self._local_sql(parts)).fetch_record_batch(10_000)
                writer = None
                try:
                    for batch in reader:
                        if writer is None:
                            writer = pq.ParquetWriter(output, batch.schema)
                        writer.write_batch(batch)
                finally:
                    if writer:
                        writer.close()
            finally:
                con.close()


def _add_radec(table: pa.Table) -> pa.Table:
    nside = 2 ** HP_HIGH_LEVEL
    ipix = table["hphigh"].to_pylist()
    ra, dec = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
    return (
        table
        .append_column("ra", pa.array(ra, type=pa.float64()))
        .append_column("dec", pa.array(dec, type=pa.float64()))
    )

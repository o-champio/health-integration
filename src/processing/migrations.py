"""Parquet schema migrations.

Each parquet file written by the pipeline is stamped with a ``schema_version``
key in its pyarrow file metadata. On load, the pipeline checks the version and
runs the appropriate migration if it's behind ``cfg.SCHEMA_VERSION``.

v1 -> v2: Oura timestamps were stored as UTC wall-clock stripped of tz info.
v2 stores them as ``cfg.LOCAL_TIMEZONE`` wall-clock stripped of tz info.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import settings as cfg

log = logging.getLogger(__name__)

# Columns that were stored as UTC-naive in v1 and need conversion to local-naive.
_V1_UTC_NAIVE_COLUMNS = {
    "daily":    ["bedtime_start", "bedtime_end"],
    "highfreq": ["timestamp"],
}


def migrate_v1_to_v2(df: pd.DataFrame, kind: Literal["daily", "highfreq"]) -> pd.DataFrame:
    """Re-localize UTC-naive columns to LOCAL_TIMEZONE-naive.

    Mathematically equivalent to what the fixed Oura client would have produced.
    pandas tz machinery handles DST transitions correctly.

    Args:
        df:   the loaded parquet DataFrame.
        kind: which column set to migrate.

    Returns:
        a new DataFrame with the same shape; the specified columns are shifted.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in _V1_UTC_NAIVE_COLUMNS.get(kind, []):
        if col not in out.columns:
            continue
        out[col] = (
            pd.to_datetime(out[col], errors="coerce")
              .dt.tz_localize("UTC")
              .dt.tz_convert(cfg.LOCAL_TIMEZONE)
              .dt.tz_localize(None)
        )
    return out


def read_schema_version(path: Path) -> int:
    """Read the schema_version stamped in a parquet file's metadata.

    Returns 1 if the file doesn't exist, has no metadata, or has no version key.
    """
    if not path.exists():
        return 1
    try:
        md = pq.read_metadata(str(path)).metadata or {}
    except Exception as exc:
        log.warning("Could not read parquet metadata for %s: %s", path.name, exc)
        return 1
    raw = md.get(b"schema_version")
    if raw is None:
        return 1
    try:
        return int(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return 1


def write_with_schema_version(df: pd.DataFrame, path: Path, version: int) -> None:
    """Write a DataFrame to parquet, stamped with schema_version in file metadata.

    Atomic: writes to a sibling .tmp.parquet first, then renames into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    new_meta = {b"schema_version": str(version).encode("ascii")}
    existing_meta = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_meta, **new_meta})
    pq.write_table(table, str(tmp))
    tmp.replace(path)

# Phase A: Timezone Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Oura client's UTC-stripping bug so all merged rows align on a single local calendar day, version the parquet schema, and migrate existing parquet files in place to preserve historical data that may be past API retention.

**Architecture:** Single helper in `oura_client.py` normalizes every API timestamp to local-naive in `cfg.LOCAL_TIMEZONE`. Parquet reads/writes are centralized through helpers that read/write a `schema_version` key in pyarrow file metadata; on load, anything below `SCHEMA_VERSION` is migrated in place (UTC-naive → local-naive on known columns) and re-saved atomically.

**Tech Stack:** Python 3.10+, pandas, pyarrow (already a transitive dep of pandas parquet I/O), pytest. Target timezone: `America/Sao_Paulo`.

**Branch:** `refactor/phase-a-timezone-correctness` (already created off `main`).

**Spec:** [docs/superpowers/specs/2026-05-16-phase-a-timezone-correctness-design.md](../specs/2026-05-16-phase-a-timezone-correctness-design.md)

---

## Task 1: Add `_to_local_naive` helper with a failing test

**Files:**
- Create: `tests/test_timezone.py`
- Modify: `src/api/oura_client.py` (add helper at the top, used by Tasks 2-4)

- [ ] **Step 1: Write the failing test**

Create `tests/test_timezone.py` with:

```python
"""Tests for timezone normalization in Oura client and migrations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api.oura_client import _to_local_naive


# ── _to_local_naive ───────────────────────────────────────────────────────────

def test_to_local_naive_shifts_iso_with_offset():
    """An ISO timestamp in UTC should be shifted to local wall-clock and stripped."""
    s = pd.Series(["2025-03-14T23:30:00Z"])
    out = _to_local_naive(s)
    assert out.dt.tz is None
    assert out.iloc[0] == pd.Timestamp("2025-03-14 20:30:00")


def test_to_local_naive_handles_offset_string():
    """An ISO timestamp with non-UTC offset should also land on local wall-clock."""
    s = pd.Series(["2025-03-14T20:30:00-03:00"])  # already São Paulo local time
    out = _to_local_naive(s)
    assert out.iloc[0] == pd.Timestamp("2025-03-14 20:30:00")


def test_to_local_naive_preserves_nat():
    """Bad inputs should become NaT, not raise."""
    s = pd.Series(["not a date", None])
    out = _to_local_naive(s)
    assert out.isna().all()


def test_to_local_naive_empty_series():
    out = _to_local_naive(pd.Series([], dtype="object"))
    assert len(out) == 0
    assert out.dt.tz is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timezone.py -v`
Expected: FAIL with `ImportError` because `_to_local_naive` doesn't exist yet.

- [ ] **Step 3: Add the helper to `src/api/oura_client.py`**

Insert just after the imports (after line 17, the `log = logging.getLogger(__name__)` line):

```python
# -- Timestamp normalization ---------------------------------------------------

def _to_local_naive(series: pd.Series) -> pd.Series:
    """Convert a series of ISO/UTC timestamps to local-naive in cfg.LOCAL_TIMEZONE.

    Oura returns timestamps with offsets (e.g. ``2025-03-14T23:30:00+00:00``).
    Stripping the offset directly leaves the wall-clock at UTC, which silently
    misaligns the data with CGM rows (already in local time). This helper does
    the conversion correctly: parse as UTC, convert to local, drop tz info.
    """
    return (
        pd.to_datetime(series, utc=True, errors="coerce")
          .dt.tz_convert(cfg.LOCAL_TIMEZONE)
          .dt.tz_localize(None)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timezone.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_timezone.py src/api/oura_client.py
git commit -m "feat(oura): add _to_local_naive helper for tz normalization"
```

---

## Task 2: Fix `get_heartrate` to use the helper

**Files:**
- Modify: `src/api/oura_client.py:135`
- Test: `tests/test_timezone.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_timezone.py`:

```python
# ── get_heartrate normalization ───────────────────────────────────────────────

from unittest.mock import patch


def _fake_oura_response(payload):
    """Mock for oura_client._get returning a fake JSON payload."""
    return payload


def test_get_heartrate_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {"timestamp": "2025-03-14T23:30:00+00:00", "bpm": 60, "source": "awake"},
            {"timestamp": "2025-03-15T02:15:00+00:00", "bpm": 55, "source": "sleep"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_heartrate("2025-03-14T00:00:00", "2025-03-15T23:59:59")

    assert df["timestamp"].dt.tz is None
    # 23:30 UTC on Mar 14 = 20:30 local on Mar 14 (São Paulo, UTC-3)
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # 02:15 UTC on Mar 15 = 23:15 local on Mar 14
    assert df.iloc[1]["timestamp"] == pd.Timestamp("2025-03-14 23:15:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timezone.py::test_get_heartrate_returns_local_naive -v`
Expected: FAIL — the current implementation produces `2025-03-14 23:30:00` (wrong) instead of `20:30:00`.

- [ ] **Step 3: Apply the fix**

In `src/api/oura_client.py`, replace line 135:

```python
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
```

with:

```python
    df["timestamp"] = _to_local_naive(df["timestamp"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timezone.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_timezone.py src/api/oura_client.py
git commit -m "fix(oura): normalize get_heartrate timestamps to local time"
```

---

## Task 3: Fix `get_sleep_sessions` to use the helper

**Files:**
- Modify: `src/api/oura_client.py:146-148`
- Test: `tests/test_timezone.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_timezone.py`:

```python
def test_get_sleep_sessions_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {
                "id": "x",
                "day": "2025-03-14",
                "bedtime_start": "2025-03-15T02:00:00+00:00",  # 23:00 local Mar 14
                "bedtime_end":   "2025-03-15T10:00:00+00:00",  # 07:00 local Mar 15
                "average_hrv": 50,
            },
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_sleep_sessions("2025-03-14", "2025-03-15")

    assert df["bedtime_start"].dt.tz is None
    assert df.iloc[0]["bedtime_start"] == pd.Timestamp("2025-03-14 23:00:00")
    assert df.iloc[0]["bedtime_end"]   == pd.Timestamp("2025-03-15 07:00:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timezone.py::test_get_sleep_sessions_returns_local_naive -v`
Expected: FAIL — current code returns `2025-03-15 02:00:00` (UTC wall-clock).

- [ ] **Step 3: Apply the fix**

In `src/api/oura_client.py`, replace lines 146-148:

```python
    for col in ["day", "bedtime_start", "bedtime_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
```

with:

```python
    for col in ["bedtime_start", "bedtime_end"]:
        if col in df.columns:
            df[col] = _to_local_naive(df[col])
    if "day" in df.columns:
        # Oura's `day` is a local-date string (e.g. "2025-03-14") — parse as naive date.
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
```

Note the split: `day` is a local date string, not a timestamp with offset, so it must NOT go through `_to_local_naive` (which would shift it by the UTC offset).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timezone.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_timezone.py src/api/oura_client.py
git commit -m "fix(oura): normalize sleep session timestamps to local time"
```

---

## Task 4: Normalize `get_workouts` start/end timestamps

**Files:**
- Modify: `src/api/oura_client.py:152-160`
- Test: `tests/test_timezone.py` (extend)

The current `get_workouts` only parses `day` and leaves `start_datetime`/`end_datetime` as raw strings. Downstream code in `workout_glucose.py` parses them defensively, but the client should produce normalized columns so future callers don't have to.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_timezone.py`:

```python
def test_get_workouts_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {
                "id": "w",
                "day": "2025-03-14",
                "activity": "running",
                "start_datetime": "2025-03-14T22:00:00+00:00",  # 19:00 local
                "end_datetime":   "2025-03-14T23:00:00+00:00",  # 20:00 local
            },
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_workouts("2025-03-14", "2025-03-14")

    assert df["start_datetime"].dt.tz is None
    assert df.iloc[0]["start_datetime"] == pd.Timestamp("2025-03-14 19:00:00")
    assert df.iloc[0]["end_datetime"]   == pd.Timestamp("2025-03-14 20:00:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timezone.py::test_get_workouts_returns_local_naive -v`
Expected: FAIL — `start_datetime` is currently a string, not a Timestamp.

- [ ] **Step 3: Apply the fix**

In `src/api/oura_client.py`, replace the body of `get_workouts` (lines 154-160) with:

```python
    raw = _get("workout", {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return df
    df = df.drop(columns=["id"], errors="ignore")
    df["day"] = pd.to_datetime(df.get("day"), errors="coerce")
    for col in ["start_datetime", "end_datetime"]:
        if col in df.columns:
            df[col] = _to_local_naive(df[col])
    return df.sort_values("day").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timezone.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_timezone.py src/api/oura_client.py
git commit -m "fix(oura): normalize workout timestamps to local time"
```

---

## Task 5: Add `SCHEMA_VERSION` constant

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Add the constant**

Open `config/settings.py`. Find the `LOCAL_TIMEZONE = "America/Sao_Paulo"` line (around line 129). Add directly below it:

```python

# -- Parquet schema versioning -------------------------------------------------
# Bumped when on-disk schema changes in a way that requires migration.
# v1: original schema (Oura timestamps stored as UTC-naive — buggy)
# v2: Oura timestamps stored as LOCAL_TIMEZONE-naive
SCHEMA_VERSION = 2
```

- [ ] **Step 2: Verify settings still import**

Run: `python -c "from config import settings; print(settings.SCHEMA_VERSION)"`
Expected: `2`

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat: add SCHEMA_VERSION constant for parquet migrations"
```

---

## Task 6: Build the migration module with failing tests

**Files:**
- Create: `src/processing/migrations.py`
- Create: `tests/test_migration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_migration.py`:

```python
"""Tests for parquet schema migrations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.processing.migrations import (
    migrate_v1_to_v2,
    read_schema_version,
    write_with_schema_version,
)
from config import settings as cfg


# ── migrate_v1_to_v2 ──────────────────────────────────────────────────────────

def test_migrate_v1_to_v2_daily_shifts_sleep_columns():
    """v1 stored sleep timestamps as UTC wall-clock; v2 stores them as local."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-03-14")],
        "bedtime_start": [pd.Timestamp("2025-03-15 02:00:00")],  # was 23:00 local
        "bedtime_end":   [pd.Timestamp("2025-03-15 10:00:00")],  # was 07:00 local
        "sleep_score":   [80],
    })
    out = migrate_v1_to_v2(df, kind="daily")
    assert out.iloc[0]["bedtime_start"] == pd.Timestamp("2025-03-14 23:00:00")
    assert out.iloc[0]["bedtime_end"]   == pd.Timestamp("2025-03-15 07:00:00")
    # Non-timestamp columns untouched.
    assert out.iloc[0]["sleep_score"] == 80


def test_migrate_v1_to_v2_highfreq_shifts_timestamp():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],  # was 20:30 local
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    out = migrate_v1_to_v2(df, kind="highfreq")
    assert out.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    assert out.iloc[0]["glucose_mgdl"] == 110.0


def test_migrate_v1_to_v2_is_NOT_idempotent_at_function_level():
    """Migration shifts every call — idempotency is enforced at the load path
    via the schema_version gate, NOT inside the function. This test documents
    that contract so callers don't bypass the gate."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-14 20:30:00")]})
    once = migrate_v1_to_v2(df, kind="highfreq")
    twice = migrate_v1_to_v2(once, kind="highfreq")
    # Two shifts of -3h each = -6h total. Intentional: never call this
    # function directly without checking schema_version first.
    assert twice.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 14:30:00")


def test_migrate_v1_to_v2_handles_missing_columns():
    """If a known column isn't present, migration is a no-op for that column."""
    df = pd.DataFrame({"date": [pd.Timestamp("2025-03-14")], "sleep_score": [80]})
    out = migrate_v1_to_v2(df, kind="daily")
    assert out.equals(df)


# ── schema_version metadata round-trip ────────────────────────────────────────

def test_write_and_read_schema_version(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "test.parquet"
    write_with_schema_version(df, path, version=2)
    assert read_schema_version(path) == 2


def test_read_schema_version_missing_metadata_returns_1(tmp_path):
    """Parquets written by older code have no metadata — treat as v1."""
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "legacy.parquet"
    df.to_parquet(path, index=False)  # no metadata
    assert read_schema_version(path) == 1


def test_read_schema_version_nonexistent_returns_1(tmp_path):
    path = tmp_path / "nope.parquet"
    assert read_schema_version(path) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_migration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.processing.migrations'`.

- [ ] **Step 3: Create the migration module**

Create `src/processing/migrations.py`:

```python
"""Parquet schema migrations.

Each parquet file written by the pipeline is stamped with a ``schema_version``
key in its pyarrow file metadata. On load, the pipeline checks the version and
runs the appropriate migration if it's behind ``cfg.SCHEMA_VERSION``.

v1 → v2: Oura timestamps were stored as UTC wall-clock stripped of tz info.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_migration.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/processing/migrations.py tests/test_migration.py
git commit -m "feat: parquet migrations module (v1→v2 tz fix + metadata helpers)"
```

---

## Task 7: Wire migration into pipeline's load/save helpers

**Files:**
- Modify: `src/processing/pipeline.py:55-69` (the existing `_load_existing` and `_save_processed`)

- [ ] **Step 1: Update the imports at the top of pipeline.py**

Find the existing imports block in `src/processing/pipeline.py` and add:

```python
from src.processing.migrations import (
    migrate_v1_to_v2,
    read_schema_version,
    write_with_schema_version,
)
```

- [ ] **Step 2: Replace `_load_existing` to handle migration**

Replace the existing `_load_existing` function (currently lines 55-62):

```python
def _load_existing(path: Path) -> pd.DataFrame | None:
    """Load an existing processed parquet file, if it exists.

    If the file's schema_version is below cfg.SCHEMA_VERSION, run the
    appropriate migration and rewrite the parquet with the new version stamp.
    Migration kind is inferred from the filename — only parquets that contain
    Oura timestamp columns need it.
    """
    if not path.exists():
        return None

    with _timed(f"Load parquet ({path.name})"):
        df = pd.read_parquet(path)
    log.info("Loaded %d rows from %s", len(df), path.name)

    version = read_schema_version(path)
    if version >= cfg.SCHEMA_VERSION:
        return df

    kind = _migration_kind_for(path)
    if kind is None:
        # No Oura timestamps in this parquet — just re-stamp.
        log.info(
            "Re-stamping %s with schema_version=%d (no migration needed).",
            path.name, cfg.SCHEMA_VERSION,
        )
        write_with_schema_version(df, path, cfg.SCHEMA_VERSION)
        return df

    log.warning(
        "Migrating %s from schema v%d to v%d (in-place).",
        path.name, version, cfg.SCHEMA_VERSION,
    )
    df = migrate_v1_to_v2(df, kind=kind)
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)
    return df


def _migration_kind_for(path: Path) -> str | None:
    """Map a parquet filename to its migration kind, or None if no migration applies."""
    name = path.name
    if name == "daily_merged.parquet":
        return "daily"
    if name == "highfreq_merged.parquet":
        return "highfreq"
    return None
```

- [ ] **Step 3: Replace `_save_processed` to stamp the schema version**

Replace the existing `_save_processed` function (currently lines 65-69):

```python
def _save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save a processed DataFrame to parquet, stamped with the current schema version."""
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)
    log.info("Saved %d rows to %s", len(df), path.name)
```

- [ ] **Step 4: Write an integration test for the load path**

Append to `tests/test_migration.py`:

```python
# ── pipeline load path integration ────────────────────────────────────────────

def test_pipeline_load_migrates_v1_parquet(tmp_path, monkeypatch):
    """Loading a v1-shaped parquet via _load_existing migrates and re-stamps it."""
    from src.processing import pipeline

    # Build a v1-style parquet: UTC-naive timestamps, no schema_version metadata.
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    path = tmp_path / "highfreq_merged.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    loaded = pipeline._load_existing(path)

    # Returned frame is migrated.
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # File on disk is rewritten with v2 stamp.
    assert read_schema_version(path) == cfg.SCHEMA_VERSION


def test_pipeline_load_skips_migration_when_current(tmp_path):
    from src.processing import pipeline

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-14 20:30:00")]})
    path = tmp_path / "highfreq_merged.parquet"
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)

    loaded = pipeline._load_existing(path)
    # No second shift was applied.
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")


def test_pipeline_load_restamps_non_oura_parquet(tmp_path):
    """A parquet with no Oura columns (e.g. glucose_readings) still gets the new stamp."""
    from src.processing import pipeline

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 10:00:00")],
        "glucose_mgdl": [110.0],
    })
    path = tmp_path / "glucose_readings.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    pipeline._load_existing(path)
    assert read_schema_version(path) == cfg.SCHEMA_VERSION
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_migration.py -v`
Expected: 9 passed (6 from Task 6 + 3 new).

Also run the existing pipeline tests to confirm no regression:

Run: `pytest tests/test_pipeline.py -v`
Expected: all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/processing/pipeline.py tests/test_migration.py
git commit -m "feat(pipeline): auto-migrate v1 parquets on load, stamp v2 on save"
```

---

## Task 8: Add merge alignment sanity check in the high-frequency builder

**Files:**
- Modify: `src/processing/pipeline.py` (the `build_highfreq_dataset` function, around line 678+)

- [ ] **Step 1: Locate the merge**

Open `src/processing/pipeline.py` and find `build_highfreq_dataset`. After the line where `result` is assigned its final merged DataFrame (just before `_save_processed(result, HIGHFREQ_PARQUET)` at line 739), we'll add a sanity check.

- [ ] **Step 2: Insert the assertion**

Insert directly before `_save_processed(result, HIGHFREQ_PARQUET)`:

```python
    # -- Sanity: post-merge timezone alignment ---------------------------------
    # All Oura/CGM timestamps must be local-naive. If a future change leaks a
    # tz-aware or UTC-naive series back in, this fires loudly here rather than
    # silently misattributing readings to the wrong calendar day.
    if not result.empty and "timestamp" in result.columns:
        ts = result["timestamp"]
        assert ts.dt.tz is None, (
            f"highfreq merge produced tz-aware timestamps (tz={ts.dt.tz}); "
            "all timestamps must be local-naive after merge."
        )
```

(We do not include a paired HR/CGM drift assertion because the current schema merges HR into the same `timestamp` column rather than a separate `hr_timestamp`. The single-column `tz is None` check is the actionable guarantee.)

- [ ] **Step 3: Run the existing pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: all pass — this is a defensive assertion, not a behavior change.

- [ ] **Step 4: Commit**

```bash
git add src/processing/pipeline.py
git commit -m "feat(pipeline): assert local-naive timestamps after highfreq merge"
```

---

## Task 9: End-to-end smoke test for daily-merge alignment

**Files:**
- Modify: `tests/test_timezone.py` (append)

This test wires together the fixed Oura client and the migration helpers to prove the full path produces aligned data on day boundaries.

- [ ] **Step 1: Write the test**

Append to `tests/test_timezone.py`:

```python
# ── End-to-end alignment ──────────────────────────────────────────────────────

def test_oura_event_near_utc_midnight_lands_on_correct_local_day():
    """An Oura HR sample at 23:30 local time on day X must land on day X locally,
    even though it is 02:30 UTC on day X+1."""
    from src.api import oura_client

    fake = {
        "data": [
            # 02:30 UTC on Mar 15 = 23:30 local on Mar 14
            {"timestamp": "2025-03-15T02:30:00+00:00", "bpm": 58, "source": "sleep"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_heartrate("2025-03-14", "2025-03-15")

    assert df.iloc[0]["timestamp"].date() == pd.Timestamp("2025-03-14").date()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_timezone.py::test_oura_event_near_utc_midnight_lands_on_correct_local_day -v`
Expected: PASS.

- [ ] **Step 3: Run the full timezone + migration test suite**

Run: `pytest tests/test_timezone.py tests/test_migration.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_timezone.py
git commit -m "test: end-to-end day-boundary alignment for Oura events"
```

---

## Task 10: Update CLAUDE.md with the timezone invariant

**Files:**
- Modify: `.claude/CLAUDE.md` (the "Domain Rules" section near the end)

- [ ] **Step 1: Edit the file**

In `.claude/CLAUDE.md`, find the bullet beginning `**Timezone normalization**:` under `## Domain Rules`. Replace it with:

```markdown
- **Timezone normalization**: All timestamps leave the API clients as local-naive in `cfg.LOCAL_TIMEZONE` (`America/Sao_Paulo`). Oura clients use the `_to_local_naive` helper in `src/api/oura_client.py`; Dexcom does the equivalent inline. LibreLink CSV timestamps are already local-naive. Daily merges key on the `day`/`date` field (Oura assigns this in the account's local tz), so a buggy timestamp client would silently misalign rows — `build_highfreq_dataset` asserts `result["timestamp"].dt.tz is None` after the merge to catch regressions.
- **Schema versioning**: Parquet files in `data/processed/` carry a `schema_version` key in pyarrow metadata. On load, `_load_existing` migrates anything below `cfg.SCHEMA_VERSION` via `src/processing/migrations.py` and rewrites the file atomically. Bump `SCHEMA_VERSION` and add a migration when on-disk shape changes.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: document tz invariant and parquet schema versioning"
```

---

## Task 11: Run the full test suite as a final verification

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: all tests pass — both the pre-existing suite and the new ones added in Tasks 1-9.

- [ ] **Step 2: If anything fails, do NOT proceed**

Diagnose the failure, fix forward (do not amend earlier commits), commit the fix, re-run.

- [ ] **Step 3: Final summary commit (optional)**

If everything is green, no additional commit is needed — the branch is ready for PR.

---

## Acceptance criteria

- All four buggy lines in `src/api/oura_client.py` (`get_heartrate`, `get_sleep_sessions`, `get_workouts`) route through `_to_local_naive`.
- `config/settings.py` exposes `SCHEMA_VERSION = 2`.
- `src/processing/migrations.py` exists and provides `migrate_v1_to_v2`, `read_schema_version`, `write_with_schema_version`.
- `src/processing/pipeline.py`'s `_load_existing` migrates v1 parquets in place; `_save_processed` stamps every write.
- `build_highfreq_dataset` asserts local-naive timestamps after merge.
- `tests/test_timezone.py` and `tests/test_migration.py` exist; full suite (`pytest tests/`) is green.
- `.claude/CLAUDE.md` documents the invariant and the schema-version mechanism.

## Out of scope (deferred)

- Pulling new fields from Oura/Dexcom (Phase B).
- Analytics consolidation, regression-page removal (Phase C).
- UI redesign + mobile (Phase D).

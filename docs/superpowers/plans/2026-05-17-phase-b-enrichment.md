# Phase B: API Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hypnogram-derived glucose-by-stage features and a rest-mode outlier flag to the daily merge, plus a `sleep_stage` column on the high-frequency dataset, with an additive v2→v3 parquet migration.

**Architecture:** A new `src/processing/stages.py` module owns the hypnogram-expansion and per-night metric logic. The Oura client gets one new method (`get_rest_mode_periods`). The pipeline's existing `_fetch_sleep_sessions` is split so the raw per-session rows (carrying `sleep_phase_5_min`) are available to the stage tagger; the existing aggregation is unchanged. Schema bumps to v3 via an additive migration; old rows get NaN/False defaults.

**Tech Stack:** Python 3.10+, pandas (with `pd.merge_asof`), pyarrow (parquet metadata), pytest. Target tz: `America/Sao_Paulo` (inherited via Phase A).

**Branch:** `feature/phase-b-enrichment` (already created off `refactor/phase-a-timezone-correctness`).

**Spec:** [docs/superpowers/specs/2026-05-17-phase-b-enrichment-design.md](../specs/2026-05-17-phase-b-enrichment-design.md)

---

## Task 1: Add `get_rest_mode_periods` to the Oura client

**Files:**
- Modify: `src/api/oura_client.py` (append at end)
- Create: `tests/test_oura_rest_mode.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_oura_rest_mode.py`:

```python
"""Tests for Oura rest_mode_period endpoint client."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api import oura_client


def test_get_rest_mode_periods_empty():
    """No periods returned -> empty DataFrame with the documented columns."""
    with patch.object(oura_client, "_get", return_value={"data": []}):
        df = oura_client.get_rest_mode_periods("2026-04-01", "2026-04-30")
    assert df.empty
    assert list(df.columns) == ["start_date", "end_date"]


def test_get_rest_mode_periods_one_period():
    """A single period yields one row with local-naive Timestamp columns."""
    fake = {
        "data": [
            {"id": "r1", "start_day": "2026-04-15", "end_day": "2026-04-17"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_rest_mode_periods("2026-04-01", "2026-04-30")

    assert len(df) == 1
    assert df["start_date"].dt.tz is None
    assert df.iloc[0]["start_date"] == pd.Timestamp("2026-04-15")
    assert df.iloc[0]["end_date"]   == pd.Timestamp("2026-04-17")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_oura_rest_mode.py -v`
Expected: FAIL with `AttributeError: module 'src.api.oura_client' has no attribute 'get_rest_mode_periods'`.

- [ ] **Step 3: Implement the function**

Append to `src/api/oura_client.py`:

```python
def get_rest_mode_periods(start_date: str, end_date: str) -> pd.DataFrame:
    """Periods where Oura was in rest mode (illness, recovery).

    Returns a DataFrame with columns ``start_date``, ``end_date`` (local-naive
    Timestamps). Used by Phase C analyses as an outlier filter — days
    overlapping any rest-mode period should typically be excluded from
    correlations.
    """
    raw = _get("rest_mode_period", {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["start_date", "end_date"])
    df = df.drop(columns=["id"], errors="ignore")
    df["start_date"] = pd.to_datetime(df.get("start_day"), errors="coerce")
    df["end_date"]   = pd.to_datetime(df.get("end_day"), errors="coerce")
    return df[["start_date", "end_date"]].sort_values("start_date").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_oura_rest_mode.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_oura_rest_mode.py src/api/oura_client.py
git commit -m "feat(oura): add get_rest_mode_periods client method"
```

---

## Task 2: Build `stages.py` module — hypnogram expansion

**Files:**
- Create: `src/processing/stages.py`
- Create: `tests/test_stages.py`

This task adds only the first stages helper. Tasks 3 and 4 add the other two.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages.py`:

```python
"""Tests for src/processing/stages.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.processing.stages import STAGE_CODES, expand_hypnogram


def test_expand_hypnogram_basic():
    """A 4-character hypnogram starting at 23:00 yields 4 rows at 5-min spacing."""
    bedtime_start = pd.Timestamp("2026-05-01 23:00:00")
    out = expand_hypnogram(bedtime_start, "1234")
    expected_times = [
        pd.Timestamp("2026-05-01 23:00:00"),
        pd.Timestamp("2026-05-01 23:05:00"),
        pd.Timestamp("2026-05-01 23:10:00"),
        pd.Timestamp("2026-05-01 23:15:00"),
    ]
    expected_stages = ["deep", "light", "rem", "awake"]
    assert out["t"].tolist() == expected_times
    assert out["stage"].tolist() == expected_stages


def test_expand_hypnogram_handles_unknown_codes():
    """Non-1234 characters become NaN stage labels but rows are still produced."""
    out = expand_hypnogram(pd.Timestamp("2026-05-01 23:00:00"), "1?2")
    assert out["stage"].tolist() == ["deep", None, "light"]


def test_expand_hypnogram_empty_string():
    """Empty hypnogram returns an empty DataFrame with the right columns."""
    out = expand_hypnogram(pd.Timestamp("2026-05-01 23:00:00"), "")
    assert out.empty
    assert list(out.columns) == ["t", "stage"]


def test_stage_codes_constant():
    """Oura encoding: 1=deep, 2=light, 3=REM, 4=awake."""
    assert STAGE_CODES == {"1": "deep", "2": "light", "3": "rem", "4": "awake"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.processing.stages'`.

- [ ] **Step 3: Create the module**

Create `src/processing/stages.py`:

```python
"""Hypnogram expansion and per-night glucose-by-stage feature derivation.

Oura returns a ``sleep_phase_5_min`` string per sleep session — one digit
per 5-minute slot starting at ``bedtime_start``. Encoding:
1=deep, 2=light, 3=REM, 4=awake.

This module turns those strings into a time-indexed stage labels, tags a
CGM frame with the stage active at each reading, and computes the per-night
metrics consumed by the daily merge.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import settings as cfg

log = logging.getLogger(__name__)

STAGE_CODES: dict[str, str] = {"1": "deep", "2": "light", "3": "rem", "4": "awake"}

_FIVE_MIN = pd.Timedelta(minutes=5)


def expand_hypnogram(bedtime_start: pd.Timestamp, code: str) -> pd.DataFrame:
    """Expand an Oura hypnogram string into one row per 5-minute slot.

    Args:
        bedtime_start: local-naive Timestamp (Phase A invariant).
        code:          string of digits, one per 5-min slot.

    Returns:
        DataFrame with columns ``t`` (Timestamp) and ``stage`` (str | None).
        Empty DataFrame with the right columns if ``code`` is empty.
    """
    if not isinstance(code, str) or not code:
        return pd.DataFrame({"t": pd.Series(dtype="datetime64[ns]"),
                             "stage": pd.Series(dtype="object")})
    times = pd.date_range(bedtime_start, periods=len(code), freq=_FIVE_MIN)
    stages = [STAGE_CODES.get(c) for c in code]
    return pd.DataFrame({"t": times, "stage": stages})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stages.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/processing/stages.py tests/test_stages.py
git commit -m "feat(stages): hypnogram expansion (5-min slot timeline)"
```

---

## Task 3: Add `tag_cgm_with_stage` to `stages.py`

**Files:**
- Modify: `src/processing/stages.py`
- Modify: `tests/test_stages.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stages.py`:

```python
from src.processing.stages import tag_cgm_with_stage


def test_tag_cgm_with_stage_inside_window():
    """CGM readings inside any sleep window get tagged; outside readings stay NaN."""
    sessions = pd.DataFrame({
        "bedtime_start": [pd.Timestamp("2026-05-01 23:00:00")],
        "sleep_phase_5_min": ["1122"],   # 23:00 deep, 23:05 deep, 23:10 light, 23:15 light
    })
    cgm = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-05-01 22:50:00"),   # before -> NaN
            pd.Timestamp("2026-05-01 23:01:00"),   # in slot 0 -> deep
            pd.Timestamp("2026-05-01 23:12:00"),   # in slot 2 -> light
            pd.Timestamp("2026-05-01 23:30:00"),   # after -> NaN
        ],
        "glucose_mgdl": [100.0, 110.0, 120.0, 130.0],
    })
    out = tag_cgm_with_stage(cgm, sessions)
    assert out["sleep_stage"].tolist() == [None, "deep", "light", None]


def test_tag_cgm_with_stage_tz_aware_input_raises():
    """Both inputs must be local-naive (Phase A invariant)."""
    sessions = pd.DataFrame({
        "bedtime_start": [pd.Timestamp("2026-05-01 23:00:00", tz="UTC")],
        "sleep_phase_5_min": ["12"],
    })
    cgm = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-05-01 23:00:00")],
        "glucose_mgdl": [100.0],
    })
    import pytest
    with pytest.raises(AssertionError):
        tag_cgm_with_stage(cgm, sessions)


def test_tag_cgm_with_stage_handles_empty_inputs():
    out_no_sessions = tag_cgm_with_stage(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-01")], "glucose_mgdl": [100.0]}),
        pd.DataFrame({"bedtime_start": pd.Series(dtype="datetime64[ns]"),
                      "sleep_phase_5_min": pd.Series(dtype="object")}),
    )
    assert out_no_sessions["sleep_stage"].isna().all()

    out_no_cgm = tag_cgm_with_stage(
        pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"),
                      "glucose_mgdl": pd.Series(dtype="float64")}),
        pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 23:00:00")],
                      "sleep_phase_5_min": ["12"]}),
    )
    assert out_no_cgm.empty
    assert "sleep_stage" in out_no_cgm.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stages.py::test_tag_cgm_with_stage_inside_window -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `tag_cgm_with_stage`**

Append to `src/processing/stages.py`:

```python
def tag_cgm_with_stage(
    cgm: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Add a ``sleep_stage`` column to a CGM frame via merge_asof.

    Args:
        cgm:      DataFrame with at least ``timestamp`` (local-naive).
        sessions: DataFrame with ``bedtime_start`` (local-naive) and
                  ``sleep_phase_5_min`` (string).

    Returns:
        ``cgm`` with one extra column ``sleep_stage``: the stage active at
        each reading's timestamp, or NaN if the reading is outside any
        sleep session (5-min tolerance).
    """
    assert cgm["timestamp"].dt.tz is None, "cgm timestamps must be local-naive"
    if "bedtime_start" in sessions.columns and not sessions.empty:
        assert sessions["bedtime_start"].dt.tz is None, "sessions bedtime_start must be local-naive"

    if cgm.empty:
        return cgm.assign(sleep_stage=pd.Series(dtype="object"))

    if sessions.empty:
        return cgm.assign(sleep_stage=pd.Series([None] * len(cgm), dtype="object"))

    frames = [
        expand_hypnogram(row["bedtime_start"], row.get("sleep_phase_5_min", ""))
        for _, row in sessions.iterrows()
    ]
    expanded = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) \
        else pd.DataFrame({"t": pd.Series(dtype="datetime64[ns]"),
                           "stage": pd.Series(dtype="object")})
    expanded = expanded.sort_values("t").reset_index(drop=True)

    cgm_sorted = cgm.sort_values("timestamp").reset_index(drop=False)
    merged = pd.merge_asof(
        cgm_sorted,
        expanded,
        left_on="timestamp",
        right_on="t",
        direction="backward",
        tolerance=_FIVE_MIN,
    )
    merged = merged.rename(columns={"stage": "sleep_stage"})
    merged = merged.set_index("index").sort_index().drop(columns=["t"])
    merged.index.name = None
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stages.py -v`
Expected: 7 passed (4 from Task 2 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/processing/stages.py tests/test_stages.py
git commit -m "feat(stages): tag CGM readings with sleep stage via merge_asof"
```

---

## Task 4: Add `per_night_glucose_by_stage` to `stages.py`

**Files:**
- Modify: `src/processing/stages.py`
- Modify: `tests/test_stages.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stages.py`:

```python
from src.processing.stages import per_night_glucose_by_stage


def test_per_night_glucose_by_stage_columns():
    """Two nights, all four stages present in both, verify the six expected columns."""
    tagged = pd.DataFrame({
        "timestamp": [
            # Night 1: 2026-05-01
            pd.Timestamp("2026-05-01 23:00:00"),  # deep, 100
            pd.Timestamp("2026-05-01 23:30:00"),  # rem, 120
            pd.Timestamp("2026-05-01 23:35:00"),  # light, 130
            pd.Timestamp("2026-05-01 23:40:00"),  # awake, 200
            # Night 2: 2026-05-02 (uses post-midnight readings)
            pd.Timestamp("2026-05-02 02:00:00"),  # deep, 180 (high)
            pd.Timestamp("2026-05-02 03:00:00"),  # rem, 150
        ],
        "glucose_mgdl": [100.0, 120.0, 130.0, 200.0, 180.0, 150.0],
        "sleep_stage": ["deep", "rem", "light", "awake", "deep", "rem"],
    })
    sessions = pd.DataFrame({
        "bedtime_start": [
            pd.Timestamp("2026-05-01 22:30:00"),
            pd.Timestamp("2026-05-02 01:00:00"),
        ],
    })
    out = per_night_glucose_by_stage(tagged, sessions)
    out = out.set_index("date")

    # Night 1 (date = 2026-05-01, the bedtime_start day)
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_deep_mean"]  == 100.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_rem_mean"]   == 120.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_light_mean"] == 130.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_awake_mean"] == 200.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_deep_minus_rem"] == -20.0
    # No high-glucose readings during deep on night 1
    assert out.loc[pd.Timestamp("2026-05-01"), "session_pct_time_high_during_deep"] == 0.0

    # Night 2 (date = 2026-05-02)
    assert out.loc[pd.Timestamp("2026-05-02"), "session_glucose_deep_mean"] == 180.0
    # 180 is NOT strictly > 180, so still 0.0 (use strict > cfg.GLUCOSE_HIGH)
    assert out.loc[pd.Timestamp("2026-05-02"), "session_pct_time_high_during_deep"] == 0.0


def test_per_night_glucose_by_stage_nan_for_missing_stage():
    """Stages absent from a night appear as NaN, not 0."""
    tagged = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-05-01 23:00:00")],
        "glucose_mgdl": [100.0],
        "sleep_stage": ["deep"],
    })
    sessions = pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 22:30:00")]})
    out = per_night_glucose_by_stage(tagged, sessions).set_index("date")
    row = out.loc[pd.Timestamp("2026-05-01")]
    assert row["session_glucose_deep_mean"] == 100.0
    assert pd.isna(row["session_glucose_rem_mean"])
    assert pd.isna(row["session_glucose_light_mean"])
    assert pd.isna(row["session_glucose_awake_mean"])
    assert pd.isna(row["session_glucose_deep_minus_rem"])  # rem missing -> diff NaN


def test_per_night_glucose_by_stage_high_threshold_uses_cfg():
    """Strict > cfg.GLUCOSE_HIGH (=180) — 181 counts, 180 does not."""
    tagged = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-05-01 23:00:00"),
            pd.Timestamp("2026-05-01 23:05:00"),
        ],
        "glucose_mgdl": [180.0, 181.0],
        "sleep_stage": ["deep", "deep"],
    })
    sessions = pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 22:30:00")]})
    out = per_night_glucose_by_stage(tagged, sessions).set_index("date")
    # 1 of 2 readings is > 180 -> 0.5
    assert out.loc[pd.Timestamp("2026-05-01"), "session_pct_time_high_during_deep"] == 0.5


def test_per_night_glucose_by_stage_empty_input():
    """Empty tagged CGM -> empty output with the right columns."""
    tagged = pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "glucose_mgdl": pd.Series(dtype="float64"),
        "sleep_stage": pd.Series(dtype="object"),
    })
    out = per_night_glucose_by_stage(tagged, pd.DataFrame())
    assert out.empty
    expected = {
        "date",
        "session_glucose_deep_mean",
        "session_glucose_light_mean",
        "session_glucose_rem_mean",
        "session_glucose_awake_mean",
        "session_glucose_deep_minus_rem",
        "session_pct_time_high_during_deep",
    }
    assert set(out.columns) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stages.py -v`
Expected: FAIL with `ImportError` on `per_night_glucose_by_stage`.

- [ ] **Step 3: Implement `per_night_glucose_by_stage`**

Append to `src/processing/stages.py`:

```python
_OUTPUT_COLUMNS = [
    "date",
    "session_glucose_deep_mean",
    "session_glucose_light_mean",
    "session_glucose_rem_mean",
    "session_glucose_awake_mean",
    "session_glucose_deep_minus_rem",
    "session_pct_time_high_during_deep",
]


def _assign_night(tagged: pd.DataFrame, sessions: pd.DataFrame) -> pd.Series:
    """Map each tagged CGM row to its sleep session's bedtime_start date.

    Uses merge_asof backward against bedtime_start so a reading at 02:00
    is assigned to the previous evening's session, not the wall-clock day.
    """
    if sessions.empty or tagged.empty:
        return pd.Series([pd.NaT] * len(tagged), index=tagged.index, dtype="datetime64[ns]")
    sess = sessions[["bedtime_start"]].sort_values("bedtime_start").reset_index(drop=True)
    sess["night"] = sess["bedtime_start"].dt.normalize()
    cgm = tagged.sort_values("timestamp").reset_index()
    merged = pd.merge_asof(
        cgm[["index", "timestamp"]],
        sess,
        left_on="timestamp",
        right_on="bedtime_start",
        direction="backward",
    )
    return merged.set_index("index")["night"].reindex(tagged.index)


def per_night_glucose_by_stage(
    tagged_cgm: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-night glucose-by-stage metrics.

    Args:
        tagged_cgm: CGM frame with ``timestamp``, ``glucose_mgdl``, ``sleep_stage``.
                    Rows where ``sleep_stage`` is NaN are dropped.
        sessions:   DataFrame with at least ``bedtime_start`` (local-naive).
                    Used to map each tagged reading to its session's night.

    Returns:
        DataFrame keyed by ``date`` (the calendar day of bedtime_start) with the
        six metric columns documented in the spec. Missing stages -> NaN.
    """
    empty_out = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    if tagged_cgm.empty:
        return empty_out

    df = tagged_cgm.dropna(subset=["sleep_stage"]).copy()
    if df.empty:
        return empty_out

    df["date"] = _assign_night(df, sessions)
    df = df.dropna(subset=["date"])
    if df.empty:
        return empty_out

    pivot = (
        df.groupby(["date", "sleep_stage"])["glucose_mgdl"]
          .mean()
          .unstack("sleep_stage")
    )
    for stage in ("deep", "light", "rem", "awake"):
        if stage not in pivot.columns:
            pivot[stage] = np.nan

    result = pd.DataFrame({
        "date": pivot.index,
        "session_glucose_deep_mean":  pivot["deep"].values,
        "session_glucose_light_mean": pivot["light"].values,
        "session_glucose_rem_mean":   pivot["rem"].values,
        "session_glucose_awake_mean": pivot["awake"].values,
    })
    result["session_glucose_deep_minus_rem"] = (
        result["session_glucose_deep_mean"] - result["session_glucose_rem_mean"]
    )

    deep_only = df[df["sleep_stage"] == "deep"]
    if deep_only.empty:
        result["session_pct_time_high_during_deep"] = np.nan
    else:
        deep_only = deep_only.assign(
            high=(deep_only["glucose_mgdl"] > cfg.GLUCOSE_HIGH).astype(float),
        )
        pct = deep_only.groupby("date")["high"].mean()
        result = result.merge(
            pct.rename("session_pct_time_high_during_deep").reset_index(),
            on="date", how="left",
        )

    return result[_OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stages.py -v`
Expected: 11 passed (7 from prior tasks + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/processing/stages.py tests/test_stages.py
git commit -m "feat(stages): per-night glucose-by-stage metrics"
```

---

## Task 5: Bump `SCHEMA_VERSION` and add v2→v3 migration

**Files:**
- Modify: `config/settings.py`
- Modify: `src/processing/migrations.py`
- Modify: `tests/test_migration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_migration.py`:

```python
# ── v2 → v3 ───────────────────────────────────────────────────────────────────

from src.processing.migrations import migrate_v2_to_v3


def test_migrate_v2_to_v3_adds_daily_columns_with_defaults():
    """v2 daily parquet lacks the new columns; v3 adds them with NaN/False defaults."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-05-01")],
        "sleep_score": [80],
    })
    out = migrate_v2_to_v3(df, kind="daily")
    assert out.iloc[0]["sleep_score"] == 80
    assert pd.isna(out.iloc[0]["session_glucose_deep_mean"])
    assert pd.isna(out.iloc[0]["session_glucose_deep_minus_rem"])
    assert pd.isna(out.iloc[0]["session_pct_time_high_during_deep"])
    assert out.iloc[0]["in_rest_mode"] is False or out.iloc[0]["in_rest_mode"] == False  # noqa: E712


def test_migrate_v2_to_v3_adds_highfreq_sleep_stage_column():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-05-01 22:00:00")],
        "glucose_mgdl": [110.0],
    })
    out = migrate_v2_to_v3(df, kind="highfreq")
    assert "sleep_stage" in out.columns
    assert pd.isna(out.iloc[0]["sleep_stage"])


def test_migrate_v2_to_v3_does_not_overwrite_existing_columns():
    """If the v2 frame already has the columns (defensive), don't blank them out."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-05-01")],
        "session_glucose_deep_mean": [99.0],  # already present somehow
    })
    out = migrate_v2_to_v3(df, kind="daily")
    assert out.iloc[0]["session_glucose_deep_mean"] == 99.0


def test_migration_chains_v1_to_v3(tmp_path):
    """A v1-shaped parquet loaded via _load_existing migrates v1→v2→v3 in one go."""
    from src.processing import pipeline

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],  # v1: UTC-naive
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    path = tmp_path / "highfreq_merged.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    loaded = pipeline._load_existing(path)

    # v1→v2 shift applied
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # v2→v3 column added
    assert "sleep_stage" in loaded.columns
    # On-disk file is now v3
    assert read_schema_version(path) == cfg.SCHEMA_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_migration.py -v`
Expected: FAIL — `migrate_v2_to_v3` doesn't exist yet.

- [ ] **Step 3: Bump `SCHEMA_VERSION`**

In `config/settings.py`, change the existing constant from:

```python
SCHEMA_VERSION = 2
```

to:

```python
SCHEMA_VERSION = 3
```

And update the comment block above it to mention v3:

```python
# v1: original schema (Oura timestamps stored as UTC-naive — buggy)
# v2: Oura timestamps stored as LOCAL_TIMEZONE-naive
# v3: adds hypnogram-derived per-night columns + sleep_stage + in_rest_mode
SCHEMA_VERSION = 3
```

- [ ] **Step 4: Add `migrate_v2_to_v3` to migrations.py**

In `src/processing/migrations.py`, add directly after the `_V1_UTC_NAIVE_COLUMNS` dict:

```python
# Columns added in v3 (additive — no data shift, just new columns with defaults).
_V2_TO_V3_NEW_COLUMNS: dict[str, dict[str, object]] = {
    "daily": {
        "session_glucose_deep_mean": np.nan,
        "session_glucose_light_mean": np.nan,
        "session_glucose_rem_mean": np.nan,
        "session_glucose_awake_mean": np.nan,
        "session_glucose_deep_minus_rem": np.nan,
        "session_pct_time_high_during_deep": np.nan,
        "in_rest_mode": False,
    },
    "highfreq": {
        "sleep_stage": pd.NA,
    },
}
```

Add the `import numpy as np` near the other imports at the top of the file (if not already present).

Then add the function (below `migrate_v1_to_v2`):

```python
def migrate_v2_to_v3(df: pd.DataFrame, kind: Literal["daily", "highfreq"]) -> pd.DataFrame:
    """Additive: add new columns with default values; leave existing data alone.

    Idempotency is enforced by the caller via ``schema_version``. The function
    itself is also defensively idempotent: pre-existing columns are not
    overwritten.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    for col, default in _V2_TO_V3_NEW_COLUMNS.get(kind, {}).items():
        if col not in out.columns:
            out[col] = default
    return out
```

- [ ] **Step 5: Chain migrations in `_load_existing`**

In `src/processing/pipeline.py`, replace the migration branch of `_load_existing` (currently a single `if/else` that calls `migrate_v1_to_v2`) with a chained walk. Find the block:

```python
    log.warning(
        "Migrating %s from schema v%d to v%d (in-place).",
        path.name, version, cfg.SCHEMA_VERSION,
    )
    df = migrate_v1_to_v2(df, kind=kind)
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)
    return df
```

Replace it with:

```python
    log.warning(
        "Migrating %s from schema v%d to v%d (in-place).",
        path.name, version, cfg.SCHEMA_VERSION,
    )
    if version < 2:
        df = migrate_v1_to_v2(df, kind=kind)
    if version < 3:
        df = migrate_v2_to_v3(df, kind=kind)
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)
    return df
```

Also update the import at the top of `pipeline.py`:

```python
from src.processing.migrations import (
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    read_schema_version,
    write_with_schema_version,
)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_migration.py tests/test_pipeline.py -v`
Expected: all green — pre-existing tests still pass, 4 new migration tests pass.

- [ ] **Step 7: Commit**

```bash
git add config/settings.py src/processing/migrations.py src/processing/pipeline.py tests/test_migration.py
git commit -m "feat: v2->v3 schema migration (additive columns)"
```

---

## Task 6: Wire raw sleep sessions through the pipeline

**Files:**
- Modify: `src/processing/pipeline.py` (`_fetch_sleep_sessions` and callers)

The current `_fetch_sleep_sessions` returns *aggregated* per-day rows; it drops `sleep_phase_5_min` and `bedtime_start`. The stage tagger needs the raw long-sleep rows. We split into two functions.

- [ ] **Step 1: Add `_fetch_sleep_sessions_raw`**

In `src/processing/pipeline.py`, insert this function just above the existing `_fetch_sleep_sessions` (around line 522):

```python
def _fetch_sleep_sessions_raw(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch the raw long-sleep session rows (one per sleep), preserving
    bedtime_start and sleep_phase_5_min for downstream stage tagging.

    Falls back to an empty DataFrame on API errors per chunk, like
    _fetch_sleep_sessions.
    """
    chunks = []
    for cs, ce in _date_chunks(start_date, end_date):
        try:
            with _timed(f"Oura sleep_sessions_raw {cs}..{ce}"):
                df = oura_client.get_sleep_sessions(cs, ce)
            if not df.empty:
                chunks.append(df)
        except Exception as exc:
            log.warning("get_sleep_sessions (%s..%s): %s", cs, ce, exc)
    if not chunks:
        return pd.DataFrame()

    sessions = pd.concat(chunks, ignore_index=True)
    if "type" in sessions.columns:
        long = sessions[sessions["type"] == "long_sleep"]
        if not long.empty:
            sessions = long
    if "total_sleep_duration" in sessions.columns:
        sessions = (
            sessions.sort_values("total_sleep_duration", ascending=False)
                    .drop_duplicates(subset=["day"], keep="first")
        )
    keep = [c for c in ["day", "bedtime_start", "bedtime_end", "sleep_phase_5_min"]
            if c in sessions.columns]
    return sessions[keep].sort_values("bedtime_start").reset_index(drop=True)
```

- [ ] **Step 2: Refactor `_fetch_sleep_sessions` to consume the raw output**

Right below the new function, replace the body of `_fetch_sleep_sessions` (currently fetches its own data). The simplest change: have it accept an optional pre-fetched DataFrame to avoid double API calls, and fall back to fetching if not provided.

Replace the existing function:

```python
def _fetch_sleep_sessions(
    start_date: str,
    end_date: str,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate detailed sleep sessions to one row per day.

    Args:
        start_date, end_date: passed through to the raw fetcher if ``raw`` is None.
        raw: optional pre-fetched DataFrame (from ``_fetch_sleep_sessions_raw``)
             to avoid a second API call.
    """
    if raw is None:
        # Need a re-fetch that keeps the aggregate columns
        chunks = []
        for cs, ce in _date_chunks(start_date, end_date):
            try:
                with _timed(f"Oura sleep_sessions {cs}..{ce}"):
                    df = oura_client.get_sleep_sessions(cs, ce)
                if not df.empty:
                    chunks.append(df)
            except Exception as exc:
                log.warning("get_sleep_sessions (%s..%s): %s", cs, ce, exc)
        if not chunks:
            return pd.DataFrame()
        sessions = pd.concat(chunks, ignore_index=True)
        if "type" in sessions.columns:
            long = sessions[sessions["type"] == "long_sleep"]
            if not long.empty:
                sessions = long
        if "total_sleep_duration" in sessions.columns:
            sessions = (
                sessions.sort_values("total_sleep_duration", ascending=False)
                .drop_duplicates(subset=["day"], keep="first")
            )
    else:
        sessions = raw

    keep_cols = ["day"]
    rename_map = {"day": "date"}
    physio_cols = {
        "average_hrv": "session_avg_hrv",
        "average_heart_rate": "session_avg_hr",
        "lowest_heart_rate": "session_lowest_hr",
        "deep_sleep_duration": "session_deep_sleep_sec",
        "rem_sleep_duration": "session_rem_sleep_sec",
        "total_sleep_duration": "session_total_sleep_sec",
        "efficiency": "session_efficiency",
        "restless_periods": "session_restless_periods",
    }
    for src, dst in physio_cols.items():
        if src in sessions.columns:
            keep_cols.append(src)
            rename_map[src] = dst

    result = sessions[keep_cols].rename(columns=rename_map).copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()

    for col in ["session_deep_sleep_sec", "session_rem_sleep_sec", "session_total_sleep_sec"]:
        min_col = col.replace("_sec", "_min")
        if col in result.columns:
            result[min_col] = (result[col] / 60).round(1)
            result = result.drop(columns=[col])

    return result.sort_values("date").reset_index(drop=True)
```

Note: when `raw` is provided, the aggregate physio columns are missing because `_fetch_sleep_sessions_raw` only keeps `day`/`bedtime_*`/`sleep_phase_5_min`. The aggregator handles this gracefully — `physio_cols` keys are guarded by `if src in sessions.columns`. So when the caller wants both aggregates and raw, they must call `get_sleep_sessions` directly with full payload (handled by the existing default-`None` `raw` path) — we **don't** chain raw → aggregator. Callers requesting both will run two fetches as before. The point of `_fetch_sleep_sessions_raw` is to make the raw data available without an aggregation pass; callers that want both still pay for two passes — acceptable for now.

- [ ] **Step 3: Run the existing pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: all 25 still pass — no behavior change for existing callers.

- [ ] **Step 4: Commit**

```bash
git add src/processing/pipeline.py
git commit -m "refactor(pipeline): split sleep-session fetch (raw + aggregator)"
```

---

## Task 7: Wire `sleep_stage` into `build_highfreq_dataset`

**Files:**
- Modify: `src/processing/pipeline.py` (`build_highfreq_dataset`)

- [ ] **Step 1: Add the stage-tagging step**

Open `src/processing/pipeline.py` and find `build_highfreq_dataset`. Right before the existing assertion `assert ts.dt.tz is None ...` (the sanity check from Phase A Task 8), insert:

```python
    # -- Tag CGM readings with sleep stage (Phase B) ---------------------------
    if not result.empty:
        from src.processing.stages import tag_cgm_with_stage
        # Pull the same date range used above for HR/CGM
        sessions = _fetch_sleep_sessions_raw(start_date, end_date)
        if not sessions.empty:
            with _timed("Tag CGM with sleep stage"):
                result = tag_cgm_with_stage(result, sessions)
        else:
            result["sleep_stage"] = pd.NA
```

Both `start_date` and `end_date` are normal `str` parameters of `build_highfreq_dataset` (see signature around line 719) — already in scope.

- [ ] **Step 2: Add the import at the top of the file**

Add to the top-level imports of `pipeline.py` (where the other `src.processing.*` imports live):

```python
from src.processing.stages import tag_cgm_with_stage
```

(And remove the inline `from src.processing.stages import tag_cgm_with_stage` inside the function.)

- [ ] **Step 3: Run the pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: 25 still pass. The pipeline tests don't fully exercise `build_highfreq_dataset` end-to-end (it requires real API calls), so this is a "no regression" check.

- [ ] **Step 4: Commit**

```bash
git add src/processing/pipeline.py
git commit -m "feat(pipeline): tag highfreq CGM rows with sleep_stage"
```

---

## Task 8: Wire glucose-by-stage features + `in_rest_mode` into `build_daily_dataset`

**Files:**
- Modify: `src/processing/pipeline.py` (`build_daily_dataset`)

- [ ] **Step 1: Insert the rest-mode fetcher near the top of `build_daily_dataset`**

In `src/processing/pipeline.py`, find `build_daily_dataset`. Locate the line where the existing daily merge has just produced the `result` DataFrame (with one row per `date`, after the `features.build_analysis_df` call or wherever the final merge lands — look for the last assignment to `result` before `_save_processed`). Insert just before the save:

```python
    # -- Glucose-by-stage per-night features (Phase B) -------------------------
    # `glucose` is already in scope (assigned at line 654 by load_glucose_only).
    if not result.empty and glucose is not None and not glucose.empty:
        from src.processing.stages import per_night_glucose_by_stage, tag_cgm_with_stage
        sessions = _fetch_sleep_sessions_raw(start_date, end_date)
        if not sessions.empty:
            tagged = tag_cgm_with_stage(glucose, sessions)
            per_night = per_night_glucose_by_stage(tagged, sessions)
            if not per_night.empty:
                result = result.merge(per_night, on="date", how="left")

    # -- Rest-mode flag (Phase B) ----------------------------------------------
    if not result.empty:
        try:
            rest = oura_client.get_rest_mode_periods(start_date, end_date)
        except Exception as exc:
            log.warning("get_rest_mode_periods (%s..%s): %s", start_date, end_date, exc)
            rest = pd.DataFrame(columns=["start_date", "end_date"])
        result["in_rest_mode"] = False
        for _, period in rest.iterrows():
            mask = (result["date"] >= period["start_date"]) & (result["date"] <= period["end_date"])
            result.loc[mask, "in_rest_mode"] = True
```

Both `start_date` and `end_date` are normal `str` parameters of `build_daily_dataset` (see signature around line 638) — already in scope by the point of insertion.

- [ ] **Step 2: Ensure six per-night columns exist even if no session data**

If no sessions are returned, `result` won't have the new columns. Add a fallback right after the rest-mode block, before `_save_processed`:

```python
    # Ensure v3 columns are always present so the parquet schema is stable.
    for col in (
        "session_glucose_deep_mean",
        "session_glucose_light_mean",
        "session_glucose_rem_mean",
        "session_glucose_awake_mean",
        "session_glucose_deep_minus_rem",
        "session_pct_time_high_during_deep",
    ):
        if col not in result.columns:
            result[col] = np.nan
    if "in_rest_mode" not in result.columns:
        result["in_rest_mode"] = False
```

Make sure `numpy as np` is already imported at the top of `pipeline.py` (it is — verify before adding).

- [ ] **Step 3: Run pipeline tests**

Run: `pytest tests/test_pipeline.py tests/test_stages.py tests/test_migration.py tests/test_oura_rest_mode.py -v`
Expected: all pre-existing + all new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/processing/pipeline.py
git commit -m "feat(pipeline): glucose-by-stage + in_rest_mode on daily merge"
```

---

## Task 9: End-to-end pipeline run + sanity verification

This task is manual — no test code. It exists to catch real-world issues the unit tests can't.

- [ ] **Step 1: Run the pipeline**

```bash
python run_pipeline.py --highfreq -v
```

Expected: pipeline completes without errors. Watch for log lines like "Tag CGM with sleep stage" and either "Loaded N rest_mode_period rows" or empty/no-error.

- [ ] **Step 2: Verify new columns are populated**

```bash
python -c "
import pandas as pd
d = pd.read_parquet('data/processed/daily_merged.parquet')
print('rows:', len(d))
new_cols = [
  'session_glucose_deep_mean', 'session_glucose_rem_mean',
  'session_glucose_deep_minus_rem', 'session_pct_time_high_during_deep',
  'in_rest_mode',
]
print(d[new_cols].describe(include='all').to_string())
print()
print('non-null counts:')
print(d[new_cols].notna().sum().to_string())
"
```

Expected:
- `non-null counts` for the four `session_glucose_*` columns should be roughly the number of nights with both Oura sleep data and CGM data (≥ 30 for the post-cutover period).
- `in_rest_mode` is a boolean — `False` for most rows, possibly some `True` if rest-mode periods exist in the account.

- [ ] **Step 3: Verify highfreq has sleep_stage**

```bash
python -c "
import pandas as pd
h = pd.read_parquet('data/processed/highfreq_merged.parquet')
print('rows:', len(h))
print('sleep_stage value_counts:')
print(h['sleep_stage'].value_counts(dropna=False))
"
```

Expected: a meaningful share of rows is `NaN` (CGM readings during the day), and the rest split across `deep`/`light`/`rem`/`awake` matching roughly the proportions from the ad-hoc analysis (light ~50%, deep ~22%, rem ~20%, awake ~8%).

- [ ] **Step 4: Verify parquet is stamped v3**

```bash
python -c "
from pathlib import Path
from src.processing.migrations import read_schema_version
for f in ['daily_merged', 'highfreq_merged']:
    p = Path('data/processed') / f'{f}.parquet'
    print(f, '->', read_schema_version(p))
"
```

Expected: both report `3`.

- [ ] **Step 5: If anything looks wrong, STOP and report**

Don't proceed to Task 10 if any of the above produces unexpected results. Diagnose, fix forward (do not amend earlier commits), commit the fix, then re-verify.

- [ ] **Step 6: Commit a record of the verification (optional)**

No commit needed for this task unless a fix was required.

---

## Task 10: Update CLAUDE.md

**Files:**
- Modify: `.claude/CLAUDE.md` (the Domain Rules section)

- [ ] **Step 1: Edit the file**

In `.claude/CLAUDE.md`, find the Domain Rules section. Add these two bullets after the existing **Schema versioning** bullet:

```markdown
- **Hypnogram-derived features** (v3): `session_glucose_*` daily columns (`deep_mean`, `light_mean`, `rem_mean`, `awake_mean`, `deep_minus_rem`, `pct_time_high_during_deep`) come from `src/processing/stages.py` operating on the `sleep_phase_5_min` strings returned by Oura's sleep endpoint. The same module also tags each high-frequency CGM row with a `sleep_stage` column. Stage encoding: 1=deep, 2=light, 3=REM, 4=awake.
- **Outlier flagging** (v3): `in_rest_mode` marks days overlapping any Oura rest-mode period (from `get_rest_mode_periods`). Phase C analyses should default to filtering these out of correlations.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: document hypnogram features and rest-mode flag (v3)"
```

---

## Task 11: Final test suite + branch summary

- [ ] **Step 1: Run the full suite**

```bash
pytest tests/ -v
```

Expected: all green. Should be on the order of 65+ tests (existing + the new test files from Tasks 1-5).

- [ ] **Step 2: If anything fails**

Diagnose, fix forward, commit, re-run.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feature/phase-b-enrichment
gh pr create --base claude/streamlit-cloud-deploy --head feature/phase-b-enrichment \
  --title "Phase B: hypnogram-derived glucose features + rest-mode flag" \
  --body "$(cat <<'EOF'
## Summary

- New `sleep_stage` column on highfreq parquet (CGM tagged with deep/light/REM/awake)
- Six new per-night columns on daily parquet:
  `session_glucose_{deep,light,rem,awake}_mean`,
  `session_glucose_deep_minus_rem`,
  `session_pct_time_high_during_deep`
- `in_rest_mode` boolean on daily parquet for outlier filtering
- Additive v2→v3 parquet migration (existing rows get NaN/False defaults)

## Validation

45-night ad-hoc test (during brainstorming) confirmed glucose differs by stage:
deep ≈ 7.5 mg/dL below REM (paired Wilcoxon p = 0.042). The hypnogram earned its keep.

## Test plan

- [x] Full pytest suite green
- [ ] User verifies: dashboard reflects the new columns after a pipeline run
- [ ] User verifies: `non-null counts` for `session_glucose_*` columns are reasonable

## Specs and plans

- Design: `docs/superpowers/specs/2026-05-17-phase-b-enrichment-design.md`
- Plan:   `docs/superpowers/plans/2026-05-17-phase-b-enrichment.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance criteria

- `src/api/oura_client.py` exposes `get_rest_mode_periods(start, end) -> DataFrame[start_date, end_date]`.
- `src/processing/stages.py` exists with `STAGE_CODES`, `expand_hypnogram`, `tag_cgm_with_stage`, `per_night_glucose_by_stage`.
- `config/settings.py`'s `SCHEMA_VERSION == 3`.
- `src/processing/migrations.py` has `migrate_v2_to_v3`; `_load_existing` walks v1 → v2 → v3.
- `build_highfreq_dataset` adds `sleep_stage`; `build_daily_dataset` adds the six glucose-by-stage columns + `in_rest_mode`.
- New test files exist: `tests/test_stages.py`, `tests/test_oura_rest_mode.py`. Existing `tests/test_migration.py` extended.
- `pytest tests/` is green.
- `.claude/CLAUDE.md` documents the new columns.

## Out of scope (deferred)

- Backfilling historical sleep_stage rows past Oura's API window.
- HRV/HR trajectory pulls.
- SpO2, vO2_max, sleep_time, tags, sessions, alerts.
- Phase C analytics decisions.

# Phase A: Data Correctness — Timezone Normalization

**Status:** Approved design — pending implementation plan
**Branch:** `refactor/phase-a-timezone-correctness`
**Date:** 2026-05-16

## Context

The health pipeline merges three data sources keyed on local calendar day or local-time timestamp:

- **Oura Ring** (API v2) — daily summaries and detailed timestamps.
- **FreeStyle LibreLink** (CSV) — historical CGM, local-naive timestamps.
- **Dexcom G7** (API v3) — current CGM, converts UTC → `LOCAL_TIMEZONE` → naive.

The Oura client strips the UTC offset *without* converting to local time, leaving Oura timestamps shifted from CGM timestamps by the local UTC offset (3 hours for `America/Sao_Paulo`). This causes silent misalignment in both the daily and high-frequency merged datasets.

## Goal

Every row in `daily_merged.parquet` and `highfreq_merged.parquet` represents a single, consistent local calendar day, with within-day timestamps that agree across sources. A regression test catches future drift.

## Non-goals

- Pulling new fields from Oura/Dexcom (Phase B).
- Analytics changes (Phase C).
- UI/mobile changes (Phase D).

## Root cause

In `src/api/oura_client.py`:

```python
# line 172
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
# line 185 — same pattern in get_sleep_sessions
df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
```

`tz_convert(None)` strips the offset and leaves the wall-clock at UTC. An Oura event at `2025-03-14T23:30:00Z` becomes naive `2025-03-14 23:30:00`, but in São Paulo that moment is `2025-03-14 20:30:00 local`. Events near midnight UTC end up re-attributed to the next calendar day.

The correct pattern (used by Dexcom, [dexcom_client.py:127-129](../../../src/api/dexcom_client.py#L127-L129)):

```python
pd.to_datetime(col, utc=True).dt.tz_convert(cfg.LOCAL_TIMEZONE).dt.tz_localize(None)
```

`get_workouts` also passes ISO strings through without normalizing — same bug, fix at the same time.

## Design

### 1. Fix the Oura client

Add a private helper in `src/api/oura_client.py`:

```python
def _to_local_naive(series: pd.Series) -> pd.Series:
    """Convert a series of ISO/UTC timestamps to local-naive in cfg.LOCAL_TIMEZONE."""
    return (
        pd.to_datetime(series, utc=True, errors="coerce")
          .dt.tz_convert(cfg.LOCAL_TIMEZONE)
          .dt.tz_localize(None)
    )
```

Replace the buggy conversions in `get_heartrate`, `get_sleep_sessions`, and `get_workouts` with `_to_local_naive(...)`. All Oura timestamps leaving the client module are local-naive in `cfg.LOCAL_TIMEZONE`.

### 2. Audit pipeline merges

`src/processing/pipeline.py`:

- **Daily merge** — keyed on Oura's `day` string field from daily summaries. Oura assigns this `day` in the account's local timezone, so it is already correct and unaffected by the timestamp bug. Add an assertion that `oura_daily.day` is dtype `object` (string), not timestamp.
- **Sleep-session merge** — the session-to-day assignment uses `bedtime_end`. Fixed transitively by the client fix.
- **High-frequency merge** — 5-min Oura HR ↔ CGM. Fixed transitively by the client fix. Add a one-line sanity check after the merge:
  ```python
  assert hf["timestamp"].dt.tz is None
  if "hr_timestamp" in hf.columns:
      drift = (hf["timestamp"] - hf["hr_timestamp"]).abs().max()
      assert drift < pd.Timedelta(minutes=5), f"HR/CGM drift {drift} exceeds 5 min"
  ```

`src/processing/workout_glucose.py` parses workout timestamps directly with utc-aware pandas; reviewed and correct — no change needed.

### 3. Schema versioning + in-place migration

Preserve historical Oura data (some of which is past the API's retention window) by migrating existing parquet rows rather than re-fetching.

**Constants** (`config/settings.py`):

```python
SCHEMA_VERSION = 2
```

**New module** `src/processing/migrations.py`:

```python
def needs_migration(parquet_path: Path) -> bool:
    """True if parquet exists but stored schema_version < SCHEMA_VERSION."""

def migrate_v1_to_v2(df: pd.DataFrame, kind: Literal["daily", "highfreq"]) -> pd.DataFrame:
    """Re-localize UTC-naive columns to LOCAL_TIMEZONE-naive."""
```

For each column listed in a `_V1_UTC_NAIVE_COLUMNS` dict per kind (sleep-session columns for "daily"; `timestamp` and any other Oura-sourced timestamp for "highfreq"):

```python
df[col] = (
    pd.to_datetime(df[col])
      .dt.tz_localize("UTC")
      .dt.tz_convert(cfg.LOCAL_TIMEZONE)
      .dt.tz_localize(None)
)
```

This is mathematically equivalent to what the fixed client would have produced. The pandas tz machinery handles any historical DST transitions correctly even though São Paulo abolished DST in 2019.

**Load path** in `pipeline.py`:

1. Read parquet metadata via `pyarrow.parquet.read_metadata(path).metadata` and extract `schema_version` (default 1 if absent).
2. If `< SCHEMA_VERSION`: log `INFO`, run `migrate_v1_to_v2`, write back atomically with the new version stamp, continue.
3. Continue incremental fetch as normal.

**Atomic write:** write to `<path>.tmp.parquet`, then `Path.replace()` to swap. Crash-safe.

**Stamping on write:** use `pyarrow.Table.replace_schema_metadata({b"schema_version": b"2", ...existing metadata...})` before `pq.write_table`. Every parquet write goes through one helper in `pipeline.py` so this is added in one place.

### 4. Tests

New file `tests/test_timezone.py`:

- `test_to_local_naive_shifts_correctly` — feeds `"2025-03-14T23:30:00Z"` through `_to_local_naive`, asserts the result is `Timestamp("2025-03-14 20:30:00")` with `tz is None`.
- `test_oura_workout_normalized` — small fixture for `get_workouts` output, asserts `start_datetime` is local-naive.
- `test_highfreq_merge_alignment` — synthetic Oura HR + CGM fixtures crossing a UTC-midnight boundary; asserts the merged rows land on the correct local day and drift assertion passes.

New file `tests/test_migration.py`:

- `test_v1_to_v2_daily` — build a small in-memory v1-shaped DataFrame with a known UTC-naive `bedtime_end`, run migration, assert the resulting timestamp falls on the expected local day.
- `test_v1_to_v2_highfreq` — same pattern for the highfreq `timestamp` column.
- `test_migration_idempotent` — running migration on already-v2 data is a no-op.
- `test_schema_version_round_trip` — write a parquet via the helper, read metadata back, assert `schema_version == 2`.

All tests are offline; no API calls.

## File-by-file change summary

| File | Change |
|---|---|
| `config/settings.py` | Add `SCHEMA_VERSION = 2`. |
| `src/api/oura_client.py` | Add `_to_local_naive` helper; use it in `get_heartrate`, `get_sleep_sessions`, `get_workouts`. |
| `src/processing/migrations.py` | **New** — migration logic + column registry. |
| `src/processing/pipeline.py` | Centralize parquet read/write through helpers that handle metadata + migration. Add merge assertions. |
| `tests/test_timezone.py` | **New**. |
| `tests/test_migration.py` | **New**. |
| `.claude/CLAUDE.md` | Add a one-line note under "Domain Rules" that all Oura timestamps leave the client as local-naive. |

## Data flow (after fix)

```
Oura API (ISO with offset)
   └─► _to_local_naive ─► local-naive ─┐
                                       ├─► pipeline merge ─► parquet (schema_version=2)
LibreLink CSV (local-naive) ───────────┤
Dexcom API (ISO) ─► local-naive ───────┘
```

On first load after upgrade: existing parquet → `needs_migration` → True → `migrate_v1_to_v2` → atomic write with v2 stamp → continue as normal.

## Risk and rollback

- **Risk:** migration applied to already-correct rows would double-shift. Mitigation: schema version gates migration; never runs twice. The `test_migration_idempotent` test enforces this.
- **Risk:** atomic write fails mid-operation. Mitigation: `.tmp.parquet` + `Path.replace` is atomic on Windows and POSIX; the original file is untouched until the rename succeeds.
- **Rollback:** the buggy v1 data is overwritten by the migration. If something goes wrong, the user can delete `data/processed/*.parquet` and rebuild from API + CSVs (Oura history loss for dates outside the API window is the only real cost, which is exactly why we preserve via migration in the first place).

## Out of scope

- Phase B: pulling additional Oura/Dexcom fields.
- Phase C: analytics consolidation.
- Phase D: UI redesign + mobile.
- Workout-glucose timestamp logic (already correct).
- Day-boundary semantics — staying with midnight-to-midnight local.

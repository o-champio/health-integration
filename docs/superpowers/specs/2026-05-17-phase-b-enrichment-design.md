# Phase B: API Enrichment — Hypnogram + Rest-Mode

**Status:** Approved scope — pending implementation plan
**Branch:** `feature/phase-b-enrichment` (off `refactor/phase-a-timezone-correctness`)
**PR target:** `claude/streamlit-cloud-deploy`
**Date:** 2026-05-17

## Context

Phase A normalized timestamps and added schema versioning. Phase B is the **enrichment** pass: pulling additional Oura/Dexcom fields that improve downstream analyses. Per the user's brief, "more data" is not the goal — anything we add must answer a concrete question.

A 45-night ad-hoc study (`2026-04-01` → `2026-05-16`) confirmed that **glucose differs by sleep stage**:

| Stage | n | mean (mg/dL) | median | CV% |
|---|---|---|---|---|
| Deep | 963 | 153 | 142 | 33.5 |
| Light | 2183 | 157 | 150 | 31.5 |
| REM | 885 | 160 | 154 | 29.9 |
| Awake (mid-night) | 377 | 163 | 162 | 32.3 |

Per-night paired deep-vs-REM: deep ≈ 7.5 mg/dL lower, **Wilcoxon p = 0.042**. The hypnogram earns its keep — pulling it enables a real Phase-C insight.

Other candidate fields (SpO2, vO2_max, sleep_time, tags, sessions, HRV/HR trajectories during sleep) were considered and **explicitly deferred** — their marginal value over what we already have is unclear without a specific analytic hook.

## Goal

Two enrichments, each grounded in a concrete downstream use:

1. **Hypnogram-derived glucose-by-stage features** on the daily merge, plus a `sleep_stage` column on the high-frequency dataset tagging each CGM reading.
2. **Rest-mode period ingestion**: a boolean `in_rest_mode` flag on the daily merge to support outlier filtering in Phase C.

A schema bump (v2 → v3) with an additive migration keeps the persisted parquets consistent without re-fetching from Oura.

## Non-goals

- Phase C (analytics consolidation / dropping the regression page).
- Phase D (UI redesign).
- Intra-night HRV/HR trajectory pulls.
- Wiring `get_daily_spo2`, `get_workouts.vO2_max`, `sleep_time`, tags, sessions, alerts.
- Re-fetching historical Oura data — the migration is additive only.

## Design

### 1. Oura client surface

Two changes in `src/api/oura_client.py`:

**a. `get_sleep_sessions` keeps additional columns.**
Currently the pipeline aggregator (`pipeline._fetch_sleep_sessions`) selects a small set of session columns and discards the rest. The hypnogram is already returned by the API (`sleep_phase_5_min`) but dropped at aggregation. The Oura client itself does not need a new method — it already returns the raw response. The change is in `pipeline._fetch_sleep_sessions` (see §3).

**b. New `get_rest_mode_periods(start_date, end_date)`.**
Calls `/usercollection/rest_mode_period` (Oura v2). Returns a tidy DataFrame with columns `start_date`, `end_date` (both local-naive). Empty DataFrame when the user has no rest-mode periods. Follows the same shape as `get_workouts` for consistency.

### 2. `src/processing/stages.py` (new module)

One responsibility: turn an Oura hypnogram string + `bedtime_start` into a stage timeline, and derive per-night metrics from a stage-tagged CGM frame.

```python
STAGE_CODES = {"1": "deep", "2": "light", "3": "rem", "4": "awake"}

def expand_hypnogram(bedtime_start: pd.Timestamp, code: str) -> pd.DataFrame:
    """One row per 5-min slot: columns t, stage."""

def tag_cgm_with_stage(cgm: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Add a `sleep_stage` column to a sorted CGM frame via merge_asof (5-min tol)."""

def per_night_glucose_by_stage(stage_tagged_cgm: pd.DataFrame) -> pd.DataFrame:
    """Per-night metrics; returned keyed by 'date' (local calendar day of bedtime_start)."""
```

Per-night columns produced (NaN if the stage didn't occur that night):

- `session_glucose_deep_mean`, `session_glucose_light_mean`, `session_glucose_rem_mean`, `session_glucose_awake_mean`
- `session_glucose_deep_minus_rem` — the validated signal
- `session_pct_time_high_during_deep` — fraction of deep-sleep minutes with glucose > `cfg.GLUCOSE_HIGH_THRESHOLD` (180 mg/dL)

These six columns are the only daily-merge additions from the hypnogram. We resist the temptation to fan out every stat × every stage — start with what the ad-hoc analysis showed mattered.

### 3. Pipeline wiring (`src/processing/pipeline.py`)

**`_fetch_sleep_sessions`:** keep `sleep_phase_5_min`, `sleep_phase_30_sec` (kept but unused for now — cheap to retain), and the day-key fields in the returned sessions DataFrame. The existing aggregate columns stay as today.

**`build_highfreq_dataset`:** after the existing CGM+HR merge, call `stages.tag_cgm_with_stage(...)` against the persisted sessions for the same date range. The `sleep_stage` column is added; CGM readings outside any sleep window get `NaN`. The Phase-A `assert result["timestamp"].dt.tz is None` continues to apply.

**`build_daily_dataset`:** after the existing daily merge, compute `stages.per_night_glucose_by_stage(...)` and left-join on `date`. Also fetch rest-mode periods and add `in_rest_mode: bool` (any rest-mode period overlapping the calendar day).

Both new joins are left joins so days without sleep-stage coverage or without CGM-during-sleep simply get NaN — no row drops.

### 4. Schema versioning + additive migration

Bump `SCHEMA_VERSION = 3` in `config/settings.py`.

The v2 → v3 migration is **additive**: it adds the new columns (`sleep_stage` for highfreq; the six glucose-by-stage columns + `in_rest_mode` for daily) with NaN/False defaults and re-stamps the file. Old historical rows that lack the source hypnogram in the cache stay NaN — no API re-fetch.

```python
_V2_TO_V3_NEW_COLUMNS = {
    "daily": {
        "session_glucose_deep_mean": np.nan,
        "session_glucose_light_mean": np.nan,
        "session_glucose_rem_mean": np.nan,
        "session_glucose_awake_mean": np.nan,
        "session_glucose_deep_minus_rem": np.nan,
        "session_pct_time_high_during_deep": np.nan,
        "in_rest_mode": False,
    },
    "highfreq": {"sleep_stage": pd.NA},
}

def migrate_v2_to_v3(df: pd.DataFrame, kind: Literal["daily", "highfreq"]) -> pd.DataFrame:
    out = df.copy()
    for col, default in _V2_TO_V3_NEW_COLUMNS.get(kind, {}).items():
        if col not in out.columns:
            out[col] = default
    return out
```

Load-path stays the same as Phase A: `_load_existing` walks v1 → v2 → v3 if needed. Migrations chain.

Once the user runs the pipeline once with the new code, the **next** incremental fetch populates the new columns going forward, while historical rows keep their NaN/False defaults. Backfilling history is not in scope.

### 5. Tests

- `tests/test_stages.py`:
  - `test_expand_hypnogram_basic` — known string + start → known timestamps + stage labels.
  - `test_expand_hypnogram_handles_unknown_codes` — non `1234` characters become NaN.
  - `test_tag_cgm_with_stage_5min_tolerance` — synthetic CGM and a single-night hypnogram; verify rows inside the window get tagged, rows outside stay NaN.
  - `test_per_night_glucose_by_stage_computes_diff` — synthetic stage-tagged frame; verify the six expected columns and that NaN appears for nights missing a stage.

- `tests/test_oura_rest_mode.py`:
  - `test_get_rest_mode_periods_empty` — mock empty response.
  - `test_get_rest_mode_periods_one_period` — mock one period; verify two local-naive dates returned.

- Extend `tests/test_migration.py`:
  - `test_migrate_v2_to_v3_adds_columns_with_defaults` — pre-migration parquet missing the new columns; post-migration has them with the documented defaults.
  - `test_migration_chains_v1_to_v3` — v1 file fully upgrades in one load.

### 6. CLAUDE.md update

Append to the "Domain Rules" section:

```markdown
- **Hypnogram-derived features**: `session_glucose_*` daily columns and the `sleep_stage` highfreq column come from `src/processing/stages.py` operating on `sleep_phase_5_min` strings from Oura. Stage encoding: 1=deep, 2=light, 3=REM, 4=awake.
- **Outlier flagging**: `in_rest_mode` marks days where Oura was in rest mode; Phase C analyses should default to filtering these out.
```

## Risk and rollback

- **Risk:** the per-night `merge_asof` mis-aligns if `sessions["bedtime_start"]` or `cgm["timestamp"]` are tz-aware. Phase A's normalization is the guarantee — Phase B adds a check assertion in `tag_cgm_with_stage`.
- **Risk:** v2 → v3 migration is double-applied. Same gate as Phase A: `read_schema_version` short-circuits when already at target.
- **Rollback:** delete `data/processed/*.parquet` and re-run pipeline. Hypnogram data is re-fetchable from Oura within the API window. Pre-window history loses the new columns but everything else regenerates.

## Out of scope (re-stated)

- SpO2, vO2_max, sleep_time, tags, sessions, alerts.
- Intra-night HRV/HR trajectory pulls.
- Backfilling historical sleep_stage / glucose-by-stage rows beyond what the cached sessions cover.
- Any Phase C analytics decisions (correlations, regression page, etc.).

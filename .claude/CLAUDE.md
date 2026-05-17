# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Health data pipeline that correlates Oura Ring biometric data (sleep, readiness, activity, HRV) with continuous glucose monitor (CGM) readings to extract actionable health insights. Includes a Streamlit dashboard with trend analysis, correlation explorer, and regression modeling.

**Data sources:**
- **Oura Ring** (API v2, OAuth2): daily summaries, detailed sleep sessions (HRV ms, deep sleep duration), 5-min heart rate, workouts. Token auto-refreshes on 401.
- **FreeStyle LibreLink** (CSV exports in `data/raw/`): ~15-min CGM readings. Used for history before `CUTOVER_DATE`.
- **Dexcom G7** (API v3, OAuth2): ~5-min CGM + logged events (carbs, insulin, exercise). Used from `CUTOVER_DATE` forward (set in `config/settings.py`).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

# OAuth (one-time, interactive -- opens browser)
python -m auth.oauth

# Launch dashboard (auto-syncs pipeline data every 12h)
streamlit run app/main.py

# Run pipeline (CLI, without dashboard)
python run_pipeline.py                            # daily dataset, full range
python run_pipeline.py --start 2025-02-01         # custom start
python run_pipeline.py --highfreq                 # also build high-freq dataset
python run_pipeline.py --no-incremental           # force full re-fetch
python run_pipeline.py -v                         # debug logging

# Run pipeline (from Python)
from src.processing.pipeline import build_daily_dataset, build_highfreq_dataset, load_glucose_only
daily = build_daily_dataset("2025-01-01", "2025-03-14")
hf = build_highfreq_dataset("2025-03-01", "2025-03-14")
glucose, stats = load_glucose_only()

# Tests (pytest, offline — no API calls required)
pytest tests/ -v
pytest tests/test_pipeline.py::test_chunks_short_range -v   # single test
pytest tests/test_performance.py -v
```

No linter or CI is configured.

## Architecture

```
config/
  settings.py          -- all paths, API URLs, scopes, thresholds, timezone
  credentials.py       -- gitignored; copy from credentials.example.py
auth/
  oauth.py             -- interactive OAuth2 code-grant flow
  tokens/              -- gitignored; oura_token.json lives here
src/
  api/
    oura_client.py     -- authenticated Oura v2 API calls, auto-refresh on 401
    libre_client.py    -- loads LibreLink CSVs, computes daily glucose stats/TIR/GMI
    dexcom_client.py   -- Dexcom G7 v3 API (EGVs + events); used from CUTOVER_DATE
  processing/
    pipeline.py        -- merges Oura daily + sleep sessions + CGM (Libre→Dexcom), incremental parquet
    features.py        -- lag features, rolling variability, derived ratios
    workout_glucose.py -- per-workout glucose response: pre/during/post windows, nadir, deltas
  models/
    analysis.py        -- Pearson/Spearman correlations, OLS regression with feature importance
app/
  main.py              -- Streamlit multi-page dashboard via st.navigation
data/
  raw/                 -- gitignored; LibreLink CSV exports
  processed/           -- gitignored; pipeline output parquet files
tests/                 -- pytest, offline (fixtures in conftest.py)
run_pipeline.py        -- CLI entry point with argparse
```

### Module dependency graph

```
config/settings.py
        |
src/api/oura_client.py + src/api/libre_client.py + src/api/dexcom_client.py
        |
src/processing/pipeline.py  (merges sources, fetches sleep sessions, saves parquet)
        |
src/processing/features.py + workout_glucose.py
        |
src/models/analysis.py      (correlation matrices, OLS regression)
        |
app/main.py                  (Streamlit st.navigation, auto-syncs via @st.cache_data)
```

### Key data flow

1. `pipeline.build_daily_dataset()` -- one row per day: glucose stats (mean, std, TIR, TBR, TAR, CV, GMI) outer-merged with Oura daily sleep/readiness/activity/stress + detailed sleep sessions (average_hrv ms, deep/rem/total sleep minutes, lowest HR, efficiency). Persists to `data/processed/daily_merged.parquet` with incremental updates.
2. `features.build_analysis_df()` -- adds physiological lag features on top of the daily dataset. Sleep metrics use shift(0) since Oura's sleep "day" already = wake day. Activity/readiness use shift(1) for next-day effects. Also adds 7-day rolling glucose, sleep-activity ratio, HRV-to-HR ratio.
3. `analysis.run_multi_target_regression()` -- OLS via statsmodels targeting glucose TIR and CV, with standardized coefficients for feature importance.

### Lag alignment logic (features.py)

- Oura's sleep "day" = the calendar day you woke up. So `sleep_score` on day X already reflects the previous night. Sleep lag features use **shift(0)** (no shift needed).
- Activity and readiness on day X = daytime metrics. Their effect on glucose manifests the next day. Activity/readiness lag features use **shift(1)**.

### Dashboard (app/main.py)

- Data auto-syncs via `@st.cache_data(ttl=43200)` (12 hours). No manual pipeline run needed.
- Multi-page via `st.navigation` / `st.Page`: Overview, Glucose Deep Dive, Lifestyle Factors, Workout Analysis, Correlation Explorer (+ regression). When adding a page, register it in the `st.navigation([...])` list near the bottom of `main.py`.
- Launch: `streamlit run app/main.py`

### Config loading

`config/settings.py` prefers environment variables (`OURA_CLIENT_ID`, `OURA_CLIENT_SECRET`, `DEXCOM_CLIENT_ID`, `DEXCOM_CLIENT_SECRET`), then a project-root `.env`, then `config/credentials.py` for local development.

### LibreLink CSV format

CSVs have a metadata row before headers (skipped via `skiprows=1`). Timestamp format: `%m-%d-%Y %I:%M %p`. Record Type 0 = historic glucose, 1 = scan, 2 = strip, 3/5 = insulin, 4 = food.

## Domain Rules

- **Timezone normalization**: All timestamps leave the API clients as local-naive in `cfg.LOCAL_TIMEZONE` (`America/Sao_Paulo`). Oura clients use the `_to_local_naive` helper in `src/api/oura_client.py`; Dexcom does the equivalent inline. LibreLink CSV timestamps are already local-naive. Daily merges key on the `day`/`date` field (Oura assigns this in the account's local tz), so a buggy timestamp client would silently misalign rows — `build_highfreq_dataset` asserts `result["timestamp"].dt.tz is None` after the merge to catch regressions.
- **Schema versioning**: Parquet files in `data/processed/` carry a `schema_version` key in pyarrow metadata. On load, `_load_existing` migrates anything below `cfg.SCHEMA_VERSION` via `src/processing/migrations.py` and rewrites the file atomically. Bump `SCHEMA_VERSION` and add a migration when on-disk shape changes.
- **Incremental updates**: `build_daily_dataset(incremental=True)` checks the last date in `data/processed/daily_merged.parquet` and only fetches new Oura data from that point. Oura API rejects queries > ~30 days (auto-chunked in `_date_chunks`).
- **Glucose thresholds**: Time-in-range = 70-180 mg/dL (configurable in `config/settings.py`). GMI formula: `3.31 + 0.02392 * mean_glucose`.
- **CGM source switch**: `CUTOVER_DATE` in `config/settings.py` is the boundary — LibreLink CSVs feed days before it, Dexcom API feeds days on/after. The pipeline concatenates both into a single glucose series.
- **Credentials**: `config/credentials.py` is gitignored. Copy from `config/credentials.example.py`. Never commit tokens or secrets.

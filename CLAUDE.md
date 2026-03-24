# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Health data pipeline that correlates Oura Ring biometric data (sleep, readiness, activity, HRV) with continuous glucose monitor (CGM) readings to extract actionable health insights. Includes a Streamlit dashboard with trend analysis, correlation explorer, and regression modeling.

**Data sources:**
- **Oura Ring** (API v2, OAuth2): daily summaries, detailed sleep sessions (HRV ms, deep sleep duration), 5-min heart rate. Token auto-refreshes on 401.
- **FreeStyle LibreLink** (CSV exports in `data/raw/`): ~15-min CGM readings. Planned migration to Dexcom API v3.

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
```

No test suite, linter, or CI exists yet.

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
  processing/
    pipeline.py        -- merges Oura daily + sleep sessions + LibreLink, incremental parquet
    features.py        -- lag features, rolling variability, derived ratios
  models/
    analysis.py        -- Pearson/Spearman correlations, OLS regression with feature importance
app/
  main.py              -- Streamlit dashboard (3 tabs: Trends, Correlations, Regression)
data/
  raw/                 -- gitignored; LibreLink CSV exports
  processed/           -- gitignored; pipeline output parquet files
run_pipeline.py        -- CLI entry point with argparse
```

### Module dependency graph

```
config/settings.py
        |
src/api/oura_client.py + src/api/libre_client.py
        |
src/processing/pipeline.py  (merges sources, fetches sleep sessions, saves parquet)
        |
src/processing/features.py  (lag features, rolling metrics, derived ratios)
        |
src/models/analysis.py      (correlation matrices, OLS regression)
        |
app/main.py                  (Streamlit dashboard, auto-syncs via @st.cache_data)
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
- Three tabs: Trend Analysis (dual-axis plotly, rolling TIR), Correlation Explorer (scatter + heatmap), Predictive Insights (OLS regression, feature importance bars).
- Launch: `streamlit run app/main.py`

### Config loading

`config/settings.py` prefers environment variables (`OURA_CLIENT_ID`, `OURA_CLIENT_SECRET`) and falls back to `config/credentials.py` for local development.

### LibreLink CSV format

CSVs have a metadata row before headers (skipped via `skiprows=1`). Timestamp format: `%m-%d-%Y %I:%M %p`. Record Type 0 = historic glucose, 1 = scan, 2 = strip, 3/5 = insulin, 4 = food.

## Domain Rules

- **Timezone normalization**: Oura returns UTC (converted to naive via `tz_convert(None)`). LibreLink timestamps are local time with no timezone. All merges must align on a consistent timezone (target: `America/Sao_Paulo` or UTC, configured in `config/settings.py`).
- **Incremental updates**: `build_daily_dataset(incremental=True)` checks the last date in `data/processed/daily_merged.parquet` and only fetches new Oura data from that point. Oura API rejects queries > ~30 days (auto-chunked in `_date_chunks`).
- **Glucose thresholds**: Time-in-range = 70-180 mg/dL (configurable in `config/settings.py`). GMI formula: `3.31 + 0.02392 * mean_glucose`.
- **Credentials**: `config/credentials.py` is gitignored. Copy from `config/credentials.example.py`. Never commit tokens or secrets.

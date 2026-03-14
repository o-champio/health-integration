# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Health data pipeline that correlates Oura Ring biometric data (sleep, readiness, activity, HRV) with continuous glucose monitor (CGM) readings to extract actionable health insights. Target output is a Streamlit dashboard.

**Data sources:**
- **Oura Ring** (API v2, OAuth2): daily summaries + 5-min heart rate. Token auto-refreshes on 401.
- **FreeStyle LibreLink** (CSV exports in `data/raw/`): ~15-min CGM readings. Planned migration to Dexcom API v3.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

# OAuth (one-time, interactive -- opens browser)
python -m auth.oauth

# Run pipeline (CLI)
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
    pipeline.py        -- merges Oura + LibreLink, lag features, incremental parquet persistence
  models/              -- (placeholder) statistical analysis, regression, correlation
app/                   -- (placeholder) Streamlit dashboard
data/
  raw/                 -- gitignored; LibreLink CSV exports
  processed/           -- gitignored; pipeline output parquet files
run_pipeline.py        -- CLI entry point with argparse
```

### Module dependency graph

```
config/credentials.py  (gitignored, holds OURA_CLIENT_ID / SECRET)
        |
config/settings.py     (loads credentials, defines all config constants)
        |
   auth/oauth.py       (interactive OAuth2 flow, saves token to auth/tokens/)
        |
src/api/oura_client.py   (authenticated API calls, auto-refresh on 401)
src/api/libre_client.py  (loads CSVs from data/raw/, computes glucose stats)
        \      /
src/processing/pipeline.py  (merges sources, adds features, saves to data/processed/)
        |
  run_pipeline.py          (CLI entry point)
```

### Key data flow

1. `build_daily_dataset()` -- one row per day: glucose stats (mean, std, TIR, TBR, TAR, CV, GMI) outer-merged with Oura daily sleep/readiness/activity/stress. Adds lag features (`*_prev_night`) and 7-day rolling glucose variability. Persists to `data/processed/daily_merged.parquet` with incremental updates.
2. `build_highfreq_dataset()` -- CGM readings asof-joined with Oura HR (nearest within +/-10 min). Persists to `data/processed/highfreq_merged.parquet`.
3. `load_glucose_only()` -- glucose data without any API calls.

### Config loading

`config/settings.py` prefers environment variables (`OURA_CLIENT_ID`, `OURA_CLIENT_SECRET`) and falls back to `config/credentials.py` for local development.

### LibreLink CSV format

CSVs have a metadata row before headers (skipped via `skiprows=1`). Timestamp format: `%m-%d-%Y %I:%M %p`. Record Type 0 = historic glucose, 1 = scan, 2 = strip, 3/5 = insulin, 4 = food.

## Domain Rules

- **Timezone normalization**: Oura returns UTC (converted to naive via `tz_convert(None)`). LibreLink timestamps are local time with no timezone. All merges must align on a consistent timezone (target: `America/Sao_Paulo` or UTC, configured in `config/settings.py`).
- **Incremental updates**: `build_daily_dataset(incremental=True)` checks the last date in `data/processed/daily_merged.parquet` and only fetches new Oura data from that point. Oura API rejects queries > ~30 days (auto-chunked in `_date_chunks`).
- **Glucose thresholds**: Time-in-range = 70-180 mg/dL (configurable in `config/settings.py`). GMI formula: `3.31 + 0.02392 * mean_glucose`.
- **Lag features**: Sleep/readiness scores are shifted by 1 day (`*_prev_night`) to capture next-day glucose impact.
- **Credentials**: `config/credentials.py` is gitignored. Copy from `config/credentials.example.py`. Never commit tokens or secrets.

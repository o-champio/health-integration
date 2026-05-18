# Health Integration

Personal health data pipeline that correlates **Oura Ring** biometrics (sleep, HRV, readiness, activity) with **continuous glucose monitor (CGM)** readings to surface actionable patterns. Includes a multi-page Streamlit dashboard with trend analysis, glucose deep-dive, workout-glucose response, correlation explorer, and OLS regression modeling.

## Data Sources

| Source | Method | Granularity |
|---|---|---|
| Oura Ring | OAuth2 API v2 | Daily summaries + 5-min HR + sleep sessions |
| FreeStyle LibreLink | CSV exports (`data/raw/`) | ~15-min CGM readings |
| Dexcom G7 | OAuth2 API v3 | ~5-min CGM + logged events (carbs, insulin, exercise) |

LibreLink is the historical source; Dexcom is used from `CUTOVER_DATE` forward.

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: .venv\Scripts\Activate.ps1  (PowerShell)
pip install -r requirements.txt
```

**Credentials** — copy the example and fill in your API keys:

```bash
cp config/credentials.example.py config/credentials.py
```

Or create a `.env` file in the project root with the same keys. See `config/credentials.example.py` for the full list (`OURA_CLIENT_ID`, `OURA_CLIENT_SECRET`, `DEXCOM_CLIENT_ID`, `DEXCOM_CLIENT_SECRET`).

**OAuth (one-time, opens browser):**

```bash
python -m auth.oauth
```

This writes tokens to `auth/tokens/` (gitignored).

## Usage

```bash
# Launch dashboard — auto-syncs pipeline data every 12 h
streamlit run app/main.py

# Run pipeline from the CLI
python run_pipeline.py                     # full range
python run_pipeline.py --start 2025-02-01  # custom start date
python run_pipeline.py --highfreq          # also build 5-min dataset
python run_pipeline.py --no-incremental    # force full re-fetch
python run_pipeline.py -v                  # debug logging
```

The dashboard auto-syncs on load (12-hour cache). You only need to run the pipeline manually when you want fresh data outside that window.

## Architecture

```
config/
  settings.py          paths, API URLs, thresholds, timezone
  credentials.py        gitignored — copy from credentials.example.py
auth/
  oauth.py             interactive OAuth2 code-grant flow (Oura + Dexcom)
  tokens/              gitignored; token JSON files live here
src/
  api/
    oura_client.py     Oura v2 API, auto-refresh on 401
    libre_client.py    LibreLink CSV loader, daily glucose stats / TIR / GMI
    dexcom_client.py   Dexcom v3 API (CGM readings, events, devices)
  processing/
    pipeline.py        merges all sources → parquet, incremental updates
    features.py        lag features, rolling metrics, derived ratios
    workout_glucose.py joins workout timestamps with CGM for response curves
  models/
    analysis.py        Pearson/Spearman correlations, OLS regression
app/
  main.py              Streamlit dashboard (6 pages via st.navigation)
data/
  raw/                 gitignored; LibreLink CSV exports
  processed/           gitignored; parquet files produced by the pipeline
run_pipeline.py        CLI entry point
```

### Dashboard Pages

| Page | What it shows |
|---|---|
| Overview | Latest metrics, period averages (TIR, CV, HRV), glucose + sleep trend |
| Glucose Deep Dive | CGM time-series, TIR donut, insulin/meal events overlay |
| Lifestyle Factors | Sleep, readiness, activity, stress vs. glucose correlations |
| Workout Analysis | Glucose response before/during/after exercise by activity type |
| Correlation Explorer | Scatter matrix + heatmap for all feature pairs |
| Regression & Insights | OLS feature importance for TIR and glucose CV |

### Key Data Flow

1. **`pipeline.build_daily_dataset()`** — one row per day: glucose stats (mean, std, TIR, TBR, TAR, CV, GMI) outer-merged with Oura daily scores + detailed sleep sessions. Saved to `data/processed/daily_merged.parquet` with incremental updates.
2. **`features.build_analysis_df()`** — adds physiological lag features. Sleep uses `shift(0)` (Oura's sleep day = wake day). Activity/readiness use `shift(1)` (next-day glucose effect).
3. **`pipeline.fetch_workouts()` + `fetch_dexcom_workouts()`** — Oura workouts merged with Dexcom exercise events (Apple Watch sessions show as `activity="unknown"`).
4. **`analysis.run_multi_target_regression()`** — OLS via statsmodels, standardized coefficients for feature importance.

## Glucose Thresholds

| Zone | Range |
|---|---|
| Time in Range (TIR) | 70–180 mg/dL |
| Time Below Range (TBR) | < 70 mg/dL |
| Time Above Range (TAR) | > 180 mg/dL |

GMI formula: `3.31 + 0.02392 × mean_glucose`. Thresholds are configurable in `config/settings.py`.

## Notes

- All timestamps are normalized to `America/Sao_Paulo` (configurable in `config/settings.py`). Oura returns UTC; LibreLink timestamps are local with no timezone.
- Oura API rejects queries longer than ~30 days — the pipeline auto-chunks date ranges.
- Never commit `config/credentials.py` or `auth/tokens/` — both are gitignored.

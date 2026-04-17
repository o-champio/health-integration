"""Performance benchmark tests.

These tests measure elapsed time for each pipeline stage and report a summary table.
They skip gracefully when the processed data / raw CSVs are not present.

Run:
    pytest tests/test_performance.py -v -s      # -s shows the timing table
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import settings as cfg

PARQUET = cfg.DATA_PROCESSED_DIR / "daily_merged.parquet"
DATA_RAW = cfg.DATA_RAW_DIR

_has_parquet = PARQUET.exists()
_has_csvs = any(DATA_RAW.glob("*.csv")) if DATA_RAW.exists() else False

# Shared timing store for the summary report
_timings: dict[str, float] = {}


def _bench(label: str, fn):
    """Run fn(), record elapsed time, return (result, elapsed)."""
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _timings[label] = elapsed
    print(f"\n  [BENCH] {label:<45} {elapsed:.3f}s")
    return result, elapsed


# ── Parquet / feature-engineering (requires processed data) ──────────────────

@pytest.mark.skipif(not _has_parquet, reason="daily_merged.parquet not present")
def test_parquet_load_speed():
    """Raw parquet read should be well under 2 s."""
    _, elapsed = _bench("pd.read_parquet(daily_merged)", lambda: pd.read_parquet(PARQUET))
    assert elapsed < 2.0, f"Parquet load took {elapsed:.3f}s — expected < 2s"


@pytest.mark.skipif(not _has_parquet, reason="daily_merged.parquet not present")
def test_build_analysis_df_speed():
    """Feature engineering on the real dataset should complete in under 1 s."""
    from src.processing.features import build_analysis_df

    daily = pd.read_parquet(PARQUET)
    _, elapsed = _bench("build_analysis_df (real data)", lambda: build_analysis_df(daily))
    assert elapsed < 1.0, f"build_analysis_df took {elapsed:.3f}s — expected < 1s"


@pytest.mark.skipif(not _has_parquet, reason="daily_merged.parquet not present")
def test_full_offline_pipeline_speed():
    """Parquet load + feature engineering + regression must all finish in under 5 s."""
    from src.models.analysis import run_multi_target_regression
    from src.processing.features import build_analysis_df, get_feature_columns

    def _run():
        daily = pd.read_parquet(PARQUET)
        df = build_analysis_df(daily)
        groups = get_feature_columns(df)
        features = groups.get("sleep_lag", []) + groups.get("activity_lag", []) + groups.get("derived", [])
        features = [f for f in features if f in df.columns][:8]  # cap at 8 for speed
        return run_multi_target_regression(df, ["glucose_tir", "glucose_cv"], features)

    _, elapsed = _bench("Full offline pipeline (load+features+regression)", _run)
    assert elapsed < 5.0, f"Full offline pipeline took {elapsed:.3f}s — expected < 5s"


# ── CSV loading (requires raw data) ──────────────────────────────────────────

@pytest.mark.skipif(not _has_csvs, reason="No raw CSV files in data/raw/")
def test_libre_load_speed():
    """Loading all LibreLink CSVs should finish in under 5 s."""
    from src.api.libre_client import load_all

    _, elapsed = _bench("libre_client.load_all (raw CSVs)", lambda: load_all(DATA_RAW))
    assert elapsed < 5.0, f"CSV load took {elapsed:.3f}s — expected < 5s"


@pytest.mark.skipif(not _has_csvs, reason="No raw CSV files in data/raw/")
def test_daily_glucose_stats_speed():
    """Per-day glucose aggregation should finish in under 2 s."""
    from src.api.libre_client import daily_glucose_stats, get_glucose_readings, load_all

    raw = load_all(DATA_RAW)
    glucose = get_glucose_readings(raw)
    _, elapsed = _bench("daily_glucose_stats", lambda: daily_glucose_stats(glucose))
    assert elapsed < 2.0, f"daily_glucose_stats took {elapsed:.3f}s — expected < 2s"


# ── Summary report (always runs last) ────────────────────────────────────────

def test_perf_report():
    """Print a summary table of all benchmark timings collected in this session."""
    if not _timings:
        pytest.skip("No benchmark timings collected (data files absent).")

    header = f"{'Stage':<48} {'Time (s)':>10}  {'Status':>10}"
    sep = "-" * len(header)
    print(f"\n\n{'='*len(header)}")
    print("  PIPELINE PERFORMANCE REPORT")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    thresholds = {
        "pd.read_parquet(daily_merged)": 2.0,
        "build_analysis_df (real data)": 1.0,
        "Full offline pipeline (load+features+regression)": 5.0,
        "libre_client.load_all (raw CSVs)": 5.0,
        "daily_glucose_stats": 2.0,
    }

    for label, elapsed in sorted(_timings.items(), key=lambda x: x[1], reverse=True):
        threshold = thresholds.get(label, float("inf"))
        status = "OK" if elapsed <= threshold else f"SLOW (>{threshold:.0f}s)"
        print(f"  {label:<46} {elapsed:>10.3f}s  {status:>10}")

    print(sep)
    total = sum(_timings.values())
    print(f"  {'TOTAL (offline stages)':<46} {total:>10.3f}s")
    print(f"{'='*len(header)}\n")

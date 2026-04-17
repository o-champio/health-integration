"""Correctness tests for pipeline, features, and analysis modules.

All tests are offline — no Oura API calls, no real files required.
Fixtures are defined in conftest.py.

Run:
    pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api import libre_client
from src.models.analysis import run_regression
from src.processing.features import build_analysis_df, get_regression_ready
from src.processing.pipeline import _date_chunks


# ── _date_chunks ──────────────────────────────────────────────────────────────

def test_chunks_short_range():
    chunks = list(_date_chunks("2025-01-01", "2025-01-05"))
    assert len(chunks) == 1
    assert chunks[0] == ("2025-01-01", "2025-01-05")


def test_chunks_exact_boundary():
    """30 days exactly must fit in a single chunk (max_days=30)."""
    chunks = list(_date_chunks("2025-01-01", "2025-01-30"))
    assert len(chunks) == 1


def test_chunks_long_range():
    """65 days → 3 chunks, no gaps and no overlaps."""
    chunks = list(_date_chunks("2025-01-01", "2025-03-06"))
    assert len(chunks) == 3

    # Rebuild all dates covered
    from datetime import datetime, timedelta
    covered = set()
    for cs, ce in chunks:
        s = datetime.strptime(cs, "%Y-%m-%d")
        e = datetime.strptime(ce, "%Y-%m-%d")
        d = s
        while d <= e:
            assert d not in covered, f"Duplicate day {d} in chunks"
            covered.add(d)
            d += timedelta(days=1)

    expected_start = datetime(2025, 1, 1)
    expected_end = datetime(2025, 3, 6)
    assert min(covered) == expected_start
    assert max(covered) == expected_end


def test_chunks_single_day():
    chunks = list(_date_chunks("2025-06-15", "2025-06-15"))
    assert len(chunks) == 1
    assert chunks[0] == ("2025-06-15", "2025-06-15")


# ── libre_client: load_csv ────────────────────────────────────────────────────

def test_load_csv_parses_timestamps(libre_csv_file):
    df = libre_client.load_csv(libre_csv_file)
    assert not df.empty
    assert pd.api.types.is_datetime64_any_dtype(df["Device Timestamp"])
    assert df["Device Timestamp"].isna().sum() == 0


def test_load_csv_header_only_returns_empty(libre_csv_header_only):
    df = libre_client.load_csv(libre_csv_header_only)
    assert df.empty


def test_get_glucose_readings_filters_type0(libre_csv_file):
    raw = libre_client.load_csv(libre_csv_file)
    glucose = libre_client.get_glucose_readings(raw)
    # type-1 scan row must be excluded
    assert len(glucose) == raw[raw["Record Type"] == 0]["Historic Glucose mg/dL"].notna().sum()
    assert "timestamp" in glucose.columns
    assert "glucose_mgdl" in glucose.columns


# ── libre_client: daily_glucose_stats ─────────────────────────────────────────

EXPECTED_STAT_COLS = [
    "date", "glucose_mean", "glucose_std", "glucose_min", "glucose_max",
    "glucose_readings", "glucose_tir", "glucose_tbr", "glucose_tar",
    "glucose_cv", "glucose_gmi",
]


def test_daily_stats_columns(daily_glucose_df):
    for col in EXPECTED_STAT_COLS:
        assert col in daily_glucose_df.columns, f"Missing column: {col}"


def test_daily_stats_tir_range(daily_glucose_df):
    """TIR must be in [0, 1] and TIR + TBR + TAR must sum to 1.0 per day."""
    assert (daily_glucose_df["glucose_tir"] >= 0).all()
    assert (daily_glucose_df["glucose_tir"] <= 1).all()
    total = (
        daily_glucose_df["glucose_tir"]
        + daily_glucose_df["glucose_tbr"]
        + daily_glucose_df["glucose_tar"]
    )
    # Values are rounded to 3 dp before storage, so allow ±0.002 tolerance
    assert ((total - 1.0).abs() <= 0.002).all(), "TIR + TBR + TAR must sum to ~1.0 for every day"


def test_daily_stats_gmi_formula(daily_glucose_df):
    """GMI = 3.31 + 0.02392 * mean_glucose (Bergenstal 2018)."""
    expected = (3.31 + 0.02392 * daily_glucose_df["glucose_mean"]).round(2)
    pd.testing.assert_series_equal(
        daily_glucose_df["glucose_gmi"].round(2),
        expected.rename("glucose_gmi"),
        check_names=False,
    )


def test_daily_stats_cv_formula(daily_glucose_df):
    """CV = std / mean, rounded to 3 dp."""
    expected = (daily_glucose_df["glucose_std"] / daily_glucose_df["glucose_mean"]).round(3)
    pd.testing.assert_series_equal(
        daily_glucose_df["glucose_cv"],
        expected.rename("glucose_cv"),
        check_names=False,
    )


# ── features: build_analysis_df ───────────────────────────────────────────────

def test_build_analysis_df_has_lag_cols(analysis_df):
    expected = [
        "prev_night_hrv",
        "prev_night_sleep_score",
        "prev_day_activity_score",
        "prev_day_steps",
        "prev_day_readiness_score",
    ]
    for col in expected:
        assert col in analysis_df.columns, f"Missing lag column: {col}"


def test_sleep_lag_is_shift0(analysis_df):
    """prev_night_hrv must equal session_avg_hrv (same-day alignment, no shift)."""
    sub = analysis_df[["session_avg_hrv", "prev_night_hrv"]].dropna()
    assert len(sub) > 0
    pd.testing.assert_series_equal(
        sub["prev_night_hrv"].reset_index(drop=True),
        sub["session_avg_hrv"].reset_index(drop=True),
        check_names=False,
    )


def test_activity_lag_is_shift1(analysis_df):
    """prev_day_activity_score[i] must equal activity_score[i-1]."""
    df = analysis_df[["activity_score", "prev_day_activity_score"]].reset_index(drop=True)
    # Compare shifted values (skip first row which becomes NaN after shift)
    for i in range(1, len(df)):
        if pd.notna(df.loc[i, "prev_day_activity_score"]) and pd.notna(df.loc[i - 1, "activity_score"]):
            assert df.loc[i, "prev_day_activity_score"] == df.loc[i - 1, "activity_score"], (
                f"Row {i}: prev_day_activity_score={df.loc[i, 'prev_day_activity_score']} "
                f"!= activity_score[{i-1}]={df.loc[i-1, 'activity_score']}"
            )


def test_rolling_glucose_7d_present(analysis_df):
    assert "glucose_mean_7d" in analysis_df.columns
    # min_periods=3, so rows 0 and 1 (only 1 and 2 observations) must be NaN
    assert pd.isna(analysis_df["glucose_mean_7d"].iloc[0])
    assert pd.isna(analysis_df["glucose_mean_7d"].iloc[1])
    # Row index 2 (3rd row) should have a value since min_periods=3
    assert pd.notna(analysis_df["glucose_mean_7d"].iloc[2])


def test_derived_ratios_finite(analysis_df):
    for col in ["hrv_hr_ratio", "sleep_activity_ratio"]:
        if col in analysis_df.columns:
            vals = analysis_df[col].dropna()
            assert np.isfinite(vals).all(), f"{col} contains non-finite values"


def test_get_regression_ready_raises_bad_target(analysis_df):
    with pytest.raises(ValueError, match="not found"):
        get_regression_ready(analysis_df, target="nonexistent_col")


def test_get_regression_ready_raises_few_rows():
    """A tiny DataFrame (< min_rows) must raise ValueError."""
    tiny = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "glucose_tir": np.random.uniform(0, 1, 5),
        "prev_night_hrv": np.random.uniform(20, 80, 5),
    })
    with pytest.raises(ValueError, match="complete rows"):
        get_regression_ready(tiny, target="glucose_tir", features=["prev_night_hrv"], min_rows=10)


# ── analysis: run_regression ──────────────────────────────────────────────────

def _make_regression_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataFrame with a known linear relationship for testing regression."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.3, n)
    y = 0.5 * x1 - 0.3 * x2 + noise + 0.7  # known coefficients
    return pd.DataFrame({"target": y, "feat_a": x1, "feat_b": x2})


def test_run_regression_synthetic():
    df = _make_regression_df(60)
    result = run_regression(df, "target", ["feat_a", "feat_b"])
    assert result.r_squared > 0
    assert result.r_squared_adj <= result.r_squared
    assert result.n_observations == 60
    assert set(result.coefficients.keys()) == {"feat_a", "feat_b"}
    assert not result.feature_importance.empty


def test_run_regression_zero_variance_dropped():
    """A constant feature should be silently dropped without raising."""
    df = _make_regression_df(60)
    df["constant"] = 5.0
    result = run_regression(df, "target", ["feat_a", "feat_b", "constant"])
    assert "constant" not in result.features
    assert result.n_observations == 60


def test_run_regression_too_few_rows():
    """Fewer than n_features + 10 observations must raise ValueError."""
    df = _make_regression_df(5)
    with pytest.raises(ValueError, match="Too few observations"):
        run_regression(df, "target", ["feat_a", "feat_b"])


# ── Incremental merge logic ───────────────────────────────────────────────────

def _run_merge(daily_glucose, oura_df, existing):
    """Reproduce the merge logic from build_daily_dataset (post-fix)."""
    if oura_df.empty:
        result = daily_glucose.copy()
    else:
        result = pd.merge(daily_glucose, oura_df, on="date", how="left")

    if existing is not None:
        glucose_col_set = set(daily_glucose.columns)
        existing_indexed = existing.set_index("date")
        for col in existing_indexed.columns:
            if col in glucose_col_set:
                continue
            if col not in result.columns:
                result[col] = pd.NA
            mask = result[col].isna()
            if mask.any():
                result.loc[mask, col] = (
                    result.loc[mask, "date"].map(existing_indexed[col])
                )

    return result.sort_values("date").reset_index(drop=True)


def test_backfill_preserves_old_oura(merge_scenario):
    """Old Oura data (days 1–60) must survive when new Oura only covers days 61–90."""
    daily_glucose, existing, new_oura = merge_scenario
    result = _run_merge(daily_glucose, new_oura, existing)

    # Days 1–60 must have Oura data backfilled from existing
    old_rows = result[result["date"] < new_oura["date"].min()]
    assert old_rows["session_avg_hrv"].notna().all(), "Old Oura data lost during merge"
    assert old_rows["sleep_score"].notna().all(), "Old sleep_score lost during merge"

    # Days 61–90 must have new Oura data
    new_rows = result[result["date"] >= new_oura["date"].min()]
    assert new_rows["session_avg_hrv"].notna().all(), "New Oura data missing"


def test_backfill_on_api_failure(merge_scenario):
    """When the API fails (oura_df empty), all Oura columns must come from existing."""
    daily_glucose, existing, _ = merge_scenario
    result = _run_merge(daily_glucose, pd.DataFrame(), existing)

    # First 60 days should have Oura data from existing
    first_60 = result.head(60)
    assert first_60["session_avg_hrv"].notna().all(), "API failure lost existing Oura data"
    assert first_60["sleep_score"].notna().all()

    # Days 61–90 have no Oura data anywhere → should be NaN
    last_30 = result.tail(30)
    assert last_30["session_avg_hrv"].isna().all()


def test_fresh_glucose_not_overwritten(merge_scenario):
    """Fresh glucose values from CSV must never be overwritten by stale parquet values."""
    daily_glucose, existing, new_oura = merge_scenario
    result = _run_merge(daily_glucose, new_oura, existing)

    # Existing has glucose_mean + 1000 (deliberately stale).
    # Result must use the fresh daily_glucose values, not the stale ones.
    merged_means = result.head(60)["glucose_mean"].values
    fresh_means = daily_glucose.head(60)["glucose_mean"].values
    np.testing.assert_array_equal(merged_means, fresh_means,
                                  err_msg="Stale glucose from parquet overwrote fresh CSV data")


def test_new_oura_columns_added(merge_scenario):
    """A column in new Oura that didn't exist in existing must appear in the result."""
    daily_glucose, existing, new_oura = merge_scenario
    new_oura = new_oura.copy()
    new_oura["brand_new_metric"] = 42.0

    result = _run_merge(daily_glucose, new_oura, existing)

    assert "brand_new_metric" in result.columns
    new_rows = result[result["date"] >= new_oura["date"].min()]
    assert (new_rows["brand_new_metric"] == 42.0).all()
    old_rows = result[result["date"] < new_oura["date"].min()]
    assert old_rows["brand_new_metric"].isna().all()

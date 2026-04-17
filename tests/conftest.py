"""Shared pytest fixtures — all purely in-memory, no API calls, no real files required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api.libre_client import daily_glucose_stats
from src.processing.features import build_analysis_df


# ── Glucose fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def glucose_df() -> pd.DataFrame:
    """90 synthetic CGM readings: 15-min intervals, glucose 60–200 mg/dL."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start, periods=90 * 4, freq="15min")  # ~360 readings / 90 days
    values = rng.uniform(60, 200, size=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "glucose_mgdl": values})


@pytest.fixture(scope="session")
def daily_glucose_df(glucose_df) -> pd.DataFrame:
    return daily_glucose_stats(glucose_df)


# ── Oura / merged fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def oura_df(daily_glucose_df) -> pd.DataFrame:
    """Synthetic Oura daily metrics aligned to the glucose date range."""
    rng = np.random.default_rng(7)
    dates = daily_glucose_df["date"].values
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "sleep_score": rng.integers(50, 100, n).astype(float),
        "readiness_score": rng.integers(50, 100, n).astype(float),
        "activity_score": rng.integers(40, 100, n).astype(float),
        "activity_steps": rng.integers(2000, 15000, n).astype(float),
        "activity_active_calories": rng.integers(100, 800, n).astype(float),
        "activity_high_activity_time": rng.integers(0, 120, n).astype(float),
        "readiness_contributors.hrv_balance": rng.integers(40, 100, n).astype(float),
        "readiness_temperature_deviation": rng.uniform(-0.5, 0.5, n),
        "stress_stress_high": rng.integers(0, 60, n).astype(float),
        "stress_recovery_high": rng.integers(0, 60, n).astype(float),
        "session_avg_hrv": rng.uniform(20, 80, n),
        "session_lowest_hr": rng.integers(45, 65, n).astype(float),
        "session_deep_sleep_min": rng.uniform(30, 120, n),
        "session_rem_sleep_min": rng.uniform(40, 130, n),
        "session_total_sleep_min": rng.uniform(300, 540, n),
        "session_efficiency": rng.integers(70, 100, n).astype(float),
        "session_restless_periods": rng.integers(0, 30, n).astype(float),
    })


@pytest.fixture(scope="session")
def merged_df(daily_glucose_df, oura_df) -> pd.DataFrame:
    return pd.merge(daily_glucose_df, oura_df, on="date", how="left")


@pytest.fixture(scope="session")
def analysis_df(merged_df) -> pd.DataFrame:
    return build_analysis_df(merged_df)


# ── LibreLink CSV fixture ─────────────────────────────────────────────────────

@pytest.fixture()
def libre_csv_file(tmp_path) -> Path:
    """Write a minimal LibreLink-format CSV to tmp_path and return its path."""
    content = (
        "Metadata row: device info\n"
        "Device Timestamp,Record Type,Historic Glucose mg/dL,Scan Glucose mg/dL,"
        "Non-numeric Rapid-Acting Insulin,Rapid-Acting Insulin (units),"
        "Non-numeric Food,Carbohydrates (grams),Non-numeric Long-Acting Insulin,"
        "Long-Acting Insulin Value (units),Notes,Strip Glucose mg/dL,Ketone mmol/L\n"
        "01-01-2025 12:00 AM,0,110,,,,,,,,,\n"
        "01-01-2025 12:15 AM,0,115,,,,,,,,,\n"
        "01-01-2025 12:30 AM,0,120,,,,,,,,,\n"
        "01-01-2025 12:45 AM,0,108,,,,,,,,,\n"
        "01-01-2025 01:00 AM,0,95,,,,,,,,,\n"
        "01-01-2025 01:15 AM,1,,112,,,,,,,,\n"   # scan reading (type 1)
        "01-01-2025 01:30 AM,0,98,,,,,,,,,\n"
        "01-02-2025 08:00 AM,0,140,,,,,,,,,\n"
        "01-02-2025 08:15 AM,0,155,,,,,,,,,\n"
        "01-02-2025 08:30 AM,0,162,,,,,,,,,\n"
    )
    p = tmp_path / "test_export.csv"
    p.write_text(content)
    return p


# ── Incremental-merge fixtures ────────────────────────────────────────────────

def _make_glucose(n_days: int, start: str = "2025-12-01") -> pd.DataFrame:
    """Build a minimal daily_glucose DataFrame with n_days rows."""
    rng = np.random.default_rng(99)
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "glucose_mean": rng.uniform(90, 140, n_days).round(2),
        "glucose_std": rng.uniform(5, 20, n_days).round(2),
        "glucose_tir": rng.uniform(0.6, 1.0, n_days).round(3),
        "glucose_cv": rng.uniform(0.04, 0.15, n_days).round(3),
    })


def _make_oura(dates: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic Oura DataFrame for the given dates."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "sleep_score": rng.integers(50, 100, n).astype(float),
        "session_avg_hrv": rng.uniform(20, 80, n).round(1),
        "activity_score": rng.integers(40, 100, n).astype(float),
    })


@pytest.fixture()
def merge_scenario():
    """Return (daily_glucose, existing_parquet, new_oura_df) for merge tests.

    - daily_glucose: 90 days (Dec 1 – Feb 28)
    - existing:      first 60 days with glucose + Oura columns
    - new_oura:      last 30 days of Oura data (days 61–90)
    """
    daily_glucose = _make_glucose(90)
    all_dates = daily_glucose["date"]

    existing_oura = _make_oura(all_dates[:60], seed=7)
    existing = pd.merge(
        _make_glucose(60),  # stale glucose (should be overwritten by fresh)
        existing_oura, on="date", how="left",
    )
    # Deliberately make stale glucose different so we can detect overwrite
    existing["glucose_mean"] = existing["glucose_mean"] + 1000

    new_oura = _make_oura(all_dates[60:], seed=42)

    return daily_glucose, existing, new_oura


@pytest.fixture()
def libre_csv_header_only(tmp_path) -> Path:
    """A CSV with metadata + header but no data rows."""
    content = (
        "Metadata row\n"
        "Device Timestamp,Record Type,Historic Glucose mg/dL,Scan Glucose mg/dL,"
        "Non-numeric Rapid-Acting Insulin,Rapid-Acting Insulin (units),"
        "Non-numeric Food,Carbohydrates (grams),Non-numeric Long-Acting Insulin,"
        "Long-Acting Insulin Value (units),Notes,Strip Glucose mg/dL,Ketone mmol/L\n"
    )
    p = tmp_path / "empty_export.csv"
    p.write_text(content)
    return p

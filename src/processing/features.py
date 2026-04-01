"""Feature engineering for glucose-biometric analysis.

Transforms the merged daily dataset (glucose stats + Oura metrics) into
an analysis-ready DataFrame with physiologically-grounded lag features,
time alignment, and derived variables.

Oura sleep "day" = the calendar day you woke up, so sleep metrics on day X
already reflect the previous night. Activity on day X = daytime activity on X.

Lag logic
---------
- prev_night_hrv, prev_night_deep_sleep, prev_night_total_sleep,
  prev_night_sleep_score  ->  shift(0): already aligned (Oura sleep day = wake day)
- prev_day_activity_score, prev_day_steps  ->  shift(1): yesterday's activity
  affects today's glucose
- prev_day_readiness_score  ->  shift(1): readiness is computed at start of day,
  but its full metabolic effect is seen the following day
"""
from __future__ import annotations

import contextlib
import logging
import time

import numpy as np
import pandas as pd

from config import settings as cfg

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _timed(label: str):
    """Log elapsed time for a feature-engineering stage at INFO level with a [PERF] prefix."""
    t0 = time.perf_counter()
    yield
    log.info("[PERF] %-45s %.3fs", label, time.perf_counter() - t0)


# -- Column discovery helpers --------------------------------------------------

# Maps from canonical feature name -> list of candidate source columns (in priority order).
# The first match found in the DataFrame is used.
_SLEEP_NIGHT_COLS = {
    "prev_night_hrv": ["session_avg_hrv"],
    "prev_night_deep_sleep_min": ["session_deep_sleep_min"],
    "prev_night_rem_sleep_min": ["session_rem_sleep_min"],
    "prev_night_total_sleep_min": ["session_total_sleep_min"],
    "prev_night_lowest_hr": ["session_lowest_hr"],
    "prev_night_sleep_score": ["sleep_score"],
    "prev_night_efficiency": ["session_efficiency"],
    "prev_night_restless": ["session_restless_periods"],
}

_PREV_DAY_COLS = {
    "prev_day_activity_score": ["activity_score"],
    "prev_day_steps": ["activity_steps"],
    "prev_day_active_calories": ["activity_active_calories"],
    "prev_day_high_activity_min": ["activity_high_activity_time"],
    "prev_day_readiness_score": ["readiness_score"],
    "prev_day_hrv_balance": ["readiness_contributors.hrv_balance"],
    "prev_day_stress_high": ["stress_stress_high"],
    "prev_day_recovery_high": ["stress_recovery_high"],
    "prev_day_body_temp_dev": ["readiness_temperature_deviation"],
}


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -- Public API ----------------------------------------------------------------

def build_analysis_df(daily: pd.DataFrame) -> pd.DataFrame:
    """Transform the merged daily dataset into an analysis-ready DataFrame.

    Adds lag features, rolling variability, and derived ratios.
    Drops rows that have no Oura data at all (ring not worn).
    """
    df = daily.copy()
    if "date" not in df.columns:
        log.warning("No 'date' column found; returning input unchanged.")
        return df

    df = df.sort_values("date").reset_index(drop=True)

    with _timed("build_analysis_df total"):
        with _timed("features: sleep night lags"):
            df = _add_sleep_night_lags(df)
        with _timed("features: prev-day lags"):
            df = _add_prev_day_lags(df)
        with _timed("features: rolling glucose"):
            df = _add_rolling_glucose(df)
        with _timed("features: derived ratios"):
            df = _add_derived_ratios(df)

    return df


def _add_sleep_night_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-night biometric features (shift=0, already aligned)."""
    for feat_name, candidates in _SLEEP_NIGHT_COLS.items():
        src = _resolve_col(df, candidates)
        if src is not None:
            df[feat_name] = df[src]
            log.debug("Mapped %s <- %s (no shift, same-day sleep alignment)", feat_name, src)
    return df


def _add_prev_day_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-day activity and readiness features (shift=1)."""
    for feat_name, candidates in _PREV_DAY_COLS.items():
        src = _resolve_col(df, candidates)
        if src is not None:
            df[feat_name] = df[src].shift(1)
            log.debug("Mapped %s <- %s.shift(1)", feat_name, src)
    return df


def _add_rolling_glucose(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day rolling glucose metrics for trend context."""
    if "glucose_mean" not in df.columns or len(df) < 3:
        return df

    df["glucose_mean_7d"] = df["glucose_mean"].rolling(7, min_periods=3).mean().round(2)
    df["glucose_cv_7d"] = df["glucose_cv"].rolling(7, min_periods=3).mean().round(3)
    if "glucose_tir" in df.columns:
        df["glucose_tir_7d"] = df["glucose_tir"].rolling(7, min_periods=3).mean().round(3)
    return df


def _add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite features with physiological rationale."""
    # Sleep-to-activity ratio: high sleep quality + high activity -> better glucose control
    if "prev_night_sleep_score" in df.columns and "prev_day_activity_score" in df.columns:
        mask = df["prev_day_activity_score"].notna() & (df["prev_day_activity_score"] > 0)
        df.loc[mask, "sleep_activity_ratio"] = (
            df.loc[mask, "prev_night_sleep_score"] / df.loc[mask, "prev_day_activity_score"]
        ).round(3)

    # HRV-to-resting-HR ratio: higher = better autonomic balance
    if "prev_night_hrv" in df.columns and "prev_night_lowest_hr" in df.columns:
        mask = df["prev_night_lowest_hr"].notna() & (df["prev_night_lowest_hr"] > 0)
        df.loc[mask, "hrv_hr_ratio"] = (
            df.loc[mask, "prev_night_hrv"] / df.loc[mask, "prev_night_lowest_hr"]
        ).round(3)

    return df


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return available feature column names grouped by category.

    Useful for the dashboard to dynamically build dropdowns.
    """
    glucose_cols = [c for c in df.columns if c.startswith("glucose_")]
    sleep_cols = [c for c in df.columns if c.startswith("prev_night_")]
    activity_cols = [c for c in df.columns if c.startswith("prev_day_")]
    derived_cols = [c for c in ["sleep_activity_ratio", "hrv_hr_ratio"] if c in df.columns]
    oura_raw = [
        c for c in df.columns
        if c.startswith(("sleep_", "readiness_", "activity_", "stress_", "session_"))
        and c not in sleep_cols + activity_cols
    ]

    return {
        "glucose": glucose_cols,
        "sleep_lag": sleep_cols,
        "activity_lag": activity_cols,
        "derived": derived_cols,
        "oura_raw": oura_raw,
    }


def get_regression_ready(
    df: pd.DataFrame,
    target: str = "glucose_tir",
    features: list[str] | None = None,
    min_rows: int = 30,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Prepare a clean DataFrame for regression (no NaNs in target+features).

    Parameters
    ----------
    df : analysis DataFrame from build_analysis_df
    target : glucose metric to predict
    features : list of predictor columns; if None, auto-selects the best
               available lag features.
    min_rows : minimum rows required after dropping NaNs.

    Returns
    -------
    (clean_df, target_col, feature_cols)

    Raises
    ------
    ValueError if fewer than min_rows remain after NaN removal.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if features is None:
        features = _auto_select_features(df)

    cols = [target] + [f for f in features if f in df.columns]
    sub = df[["date"] + cols].dropna(subset=cols)

    if len(sub) < min_rows:
        raise ValueError(
            f"Only {len(sub)} complete rows for target='{target}' "
            f"(need {min_rows}). Features: {cols}"
        )

    log.info(
        "Regression-ready: %d rows, target=%s, %d features.",
        len(sub), target, len(cols) - 1,
    )
    return sub, target, [f for f in features if f in df.columns]


def _auto_select_features(df: pd.DataFrame) -> list[str]:
    """Pick the best available lag features for regression."""
    priority = [
        "prev_night_hrv",
        "prev_night_deep_sleep_min",
        "prev_night_total_sleep_min",
        "prev_night_sleep_score",
        "prev_day_activity_score",
        "prev_day_steps",
        "prev_day_readiness_score",
        "prev_day_hrv_balance",
        "prev_day_body_temp_dev",
        "prev_night_lowest_hr",
        "prev_day_stress_high",
        "prev_day_recovery_high",
        "sleep_activity_ratio",
        "hrv_hr_ratio",
    ]
    return [f for f in priority if f in df.columns]

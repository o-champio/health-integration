"""Workout-glucose analysis: per-workout glucose response metrics.

Joins Oura workout timestamps with CGM readings to compute glucose
behaviour before, during, and after each exercise session.

Excluded activities: walking, houseWork (low signal-to-noise).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import settings as cfg

log = logging.getLogger(__name__)

_EXCLUDE_ACTIVITIES = {"walking", "houseWork"}
_ALWAYS_INCLUDE = {"running", "strengthTraining", "cycling"}
_MIN_WORKOUTS_PER_TYPE = 3


# -- Helpers -------------------------------------------------------------------

def _to_naive_local(ts_series: pd.Series) -> pd.Series:
    """Convert timezone-aware ISO strings to naive local timestamps.

    Oura workout timestamps are ISO-8601 with offset (e.g. -05:00).
    Glucose timestamps are naive local time. This converts Oura times
    to the same representation via LOCAL_TIMEZONE.
    """
    aware = pd.to_datetime(ts_series, utc=True)
    return aware.dt.tz_convert(cfg.LOCAL_TIMEZONE).dt.tz_localize(None)


def _classify_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    return "evening"


def _window_glucose(glucose: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return glucose readings within [start, end]."""
    return glucose[(glucose["timestamp"] >= start) & (glucose["timestamp"] <= end)]


# -- Core analysis -------------------------------------------------------------

def build_workout_glucose_df(
    glucose: pd.DataFrame,
    workouts: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-workout DataFrame with glucose metrics in multiple windows.

    For each workout computes:
    - pre_avg: mean glucose in the 30 min before workout start
    - during_avg: mean glucose during the workout
    - post_60_avg: mean glucose in the 60 min after workout end
    - delta_during: during_avg - pre_avg
    - delta_post: post_60_avg - pre_avg
    - nadir_post_120: lowest glucose reading in 2 h after workout end
    - nadir_time_min: minutes from workout end to the nadir reading
    - time_of_day: morning / afternoon / evening
    """
    if workouts.empty or glucose.empty:
        return pd.DataFrame()

    wk = workouts[~workouts["activity"].isin(_EXCLUDE_ACTIVITIES)].copy()
    if wk.empty:
        return pd.DataFrame()

    wk["start_local"] = _to_naive_local(wk["start_datetime"])
    wk["end_local"] = _to_naive_local(wk["end_datetime"])

    glucose_sorted = glucose.sort_values("timestamp")

    rows: list[dict] = []
    for _, w in wk.iterrows():
        start = w["start_local"]
        end = w["end_local"]

        pre = _window_glucose(glucose_sorted, start - pd.Timedelta(minutes=30), start)
        during = _window_glucose(glucose_sorted, start, end)
        post_60 = _window_glucose(glucose_sorted, end, end + pd.Timedelta(minutes=60))
        post_120 = _window_glucose(glucose_sorted, end, end + pd.Timedelta(minutes=120))

        pre_avg = pre["glucose_mgdl"].mean() if len(pre) else np.nan
        during_avg = during["glucose_mgdl"].mean() if len(during) else np.nan
        post_60_avg = post_60["glucose_mgdl"].mean() if len(post_60) else np.nan

        nadir_val = np.nan
        nadir_time = np.nan
        if len(post_120):
            nadir_idx = post_120["glucose_mgdl"].idxmin()
            nadir_val = post_120.loc[nadir_idx, "glucose_mgdl"]
            nadir_time = (post_120.loc[nadir_idx, "timestamp"] - end).total_seconds() / 60

        rows.append({
            "activity": w["activity"],
            "day": w["day"],
            "start_local": start,
            "end_local": end,
            "duration_min": round((end - start).total_seconds() / 60, 1),
            "calories": w.get("calories", np.nan),
            "time_of_day": _classify_time_of_day(start.hour),
            "pre_avg": round(pre_avg, 1) if pd.notna(pre_avg) else np.nan,
            "during_avg": round(during_avg, 1) if pd.notna(during_avg) else np.nan,
            "post_60_avg": round(post_60_avg, 1) if pd.notna(post_60_avg) else np.nan,
            "delta_during": round(during_avg - pre_avg, 1) if pd.notna(during_avg) and pd.notna(pre_avg) else np.nan,
            "delta_post": round(post_60_avg - pre_avg, 1) if pd.notna(post_60_avg) and pd.notna(pre_avg) else np.nan,
            "nadir_post_120": round(nadir_val, 1) if pd.notna(nadir_val) else np.nan,
            "nadir_time_min": round(nadir_time, 0) if pd.notna(nadir_time) else np.nan,
            "n_readings_pre": len(pre),
            "n_readings_during": len(during),
            "n_readings_post": len(post_60),
        })

    result = pd.DataFrame(rows)

    # Drop activity types with fewer than _MIN_WORKOUTS_PER_TYPE workouts
    # (pinned activities in _ALWAYS_INCLUDE are kept regardless of count)
    if not result.empty:
        counts = result["activity"].value_counts()
        keep = counts[counts >= _MIN_WORKOUTS_PER_TYPE].index.union(_ALWAYS_INCLUDE)
        dropped = set(result["activity"].unique()) - set(keep)
        if dropped:
            log.info("Dropping activity types with n<%d: %s", _MIN_WORKOUTS_PER_TYPE, dropped)
        result = result[result["activity"].isin(keep)]

    log.info("Built workout-glucose dataset: %d workouts with glucose overlap.", len(result))
    return result.sort_values("start_local").reset_index(drop=True)


def glucose_response_curve(
    glucose: pd.DataFrame,
    workouts: pd.DataFrame,
    window_before: int = 30,
    window_after: int = 90,
    step: int = 5,
) -> pd.DataFrame:
    """Build an aligned glucose trajectory from -window_before to +window_after.

    Returns a long-form DataFrame with columns:
        relative_min, glucose_mgdl, glucose_delta (vs T0), activity

    Each workout's glucose is resampled onto a regular grid (step-min intervals)
    relative to workout start (T0). glucose_delta = glucose - glucose at T0.
    The result is averaged by activity type for plotting.
    """
    if workouts.empty or glucose.empty:
        return pd.DataFrame()

    wk = workouts[~workouts["activity"].isin(_EXCLUDE_ACTIVITIES)].copy()
    if wk.empty:
        return pd.DataFrame()

    # Drop activity types with fewer than _MIN_WORKOUTS_PER_TYPE workouts
    # (pinned activities in _ALWAYS_INCLUDE are kept regardless of count)
    counts = wk["activity"].value_counts()
    keep = counts[counts >= _MIN_WORKOUTS_PER_TYPE].index.union(_ALWAYS_INCLUDE)
    wk = wk[wk["activity"].isin(keep)]
    if wk.empty:
        return pd.DataFrame()

    wk["start_local"] = _to_naive_local(wk["start_datetime"])
    wk["end_local"] = _to_naive_local(wk["end_datetime"])

    glucose_sorted = glucose.sort_values("timestamp")
    grid = list(range(-window_before, window_after + 1, step))

    all_traces: list[pd.DataFrame] = []

    for idx, (_, w) in enumerate(wk.iterrows()):
        t0 = w["start_local"]
        window = _window_glucose(
            glucose_sorted,
            t0 - pd.Timedelta(minutes=window_before + step),
            t0 + pd.Timedelta(minutes=window_after + step),
        )
        if len(window) < 3:
            continue

        # Find glucose nearest to T0 for baseline
        diffs = (window["timestamp"] - t0).abs()
        baseline = window.loc[diffs.idxmin(), "glucose_mgdl"]

        points = []
        for rel_min in grid:
            target_time = t0 + pd.Timedelta(minutes=rel_min)
            tolerance = pd.Timedelta(minutes=step / 2 + 1)
            nearby = window[
                (window["timestamp"] >= target_time - tolerance)
                & (window["timestamp"] <= target_time + tolerance)
            ]
            if nearby.empty:
                continue
            closest_idx = (nearby["timestamp"] - target_time).abs().idxmin()
            val = nearby.loc[closest_idx, "glucose_mgdl"]
            points.append({
                "relative_min": rel_min,
                "glucose_mgdl": val,
                "glucose_delta": round(val - baseline, 1),
                "activity": w["activity"],
                "workout_idx": idx,
            })

        if points:
            all_traces.append(pd.DataFrame(points))

    if not all_traces:
        return pd.DataFrame()

    return pd.concat(all_traces, ignore_index=True)


def workout_summary_by_type(wg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-workout metrics by activity type."""
    if wg.empty:
        return pd.DataFrame()

    agg = (
        wg.groupby("activity")
        .agg(
            workouts=("activity", "size"),
            avg_duration=("duration_min", "mean"),
            avg_calories=("calories", "mean"),
            avg_pre=("pre_avg", "mean"),
            avg_delta_during=("delta_during", "mean"),
            avg_delta_post=("delta_post", "mean"),
            avg_nadir=("nadir_post_120", "mean"),
            avg_nadir_time=("nadir_time_min", "mean"),
        )
        .round(1)
        .sort_values("workouts", ascending=False)
        .reset_index()
    )
    return agg

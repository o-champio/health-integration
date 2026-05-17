"""Hypnogram expansion.

Oura returns a ``sleep_phase_5_min`` string per sleep session — one digit
per 5-minute slot starting at ``bedtime_start``. Encoding:
1=deep, 2=light, 3=REM, 4=awake.

This module expands such strings into a time-indexed stage timeline.
Subsequent helpers (CGM tagging, per-night metrics) will be added in
follow-up tasks and will document themselves.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import settings as cfg

log = logging.getLogger(__name__)

STAGE_CODES: dict[str, str] = {"1": "deep", "2": "light", "3": "rem", "4": "awake"}

_FIVE_MIN = pd.Timedelta(minutes=5)


def tag_cgm_with_stage(
    cgm: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Add a ``sleep_stage`` column to a CGM frame via merge_asof.

    Args:
        cgm:      DataFrame with at least ``timestamp`` (local-naive).
        sessions: DataFrame with ``bedtime_start`` (local-naive) and
                  ``sleep_phase_5_min`` (string).

    Returns:
        ``cgm`` with one extra column ``sleep_stage``: the stage active at
        each reading's timestamp, or NaN if the reading is outside any
        sleep session (5-min tolerance).
    """
    assert cgm["timestamp"].dt.tz is None, "cgm timestamps must be local-naive"
    if "bedtime_start" in sessions.columns and not sessions.empty:
        assert sessions["bedtime_start"].dt.tz is None, "sessions bedtime_start must be local-naive"

    if cgm.empty:
        return cgm.assign(sleep_stage=pd.Series(dtype="object"))

    if sessions.empty:
        return cgm.assign(sleep_stage=pd.Series([None] * len(cgm), dtype="object"))

    frames = [
        expand_hypnogram(row["bedtime_start"], row.get("sleep_phase_5_min", ""))
        for _, row in sessions.iterrows()
    ]
    expanded = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) \
        else pd.DataFrame({"t": pd.Series(dtype="datetime64[ns]"),
                           "stage": pd.Series(dtype="object")})
    expanded = expanded.sort_values("t").reset_index(drop=True)

    cgm_sorted = cgm.sort_values("timestamp").reset_index(drop=False)
    merged = pd.merge_asof(
        cgm_sorted,
        expanded,
        left_on="timestamp",
        right_on="t",
        direction="backward",
        tolerance=_FIVE_MIN,
    )
    merged = merged.rename(columns={"stage": "sleep_stage"})
    merged = merged.set_index("index").sort_index().drop(columns=["t"])
    merged.index.name = None
    # Convert float NaN (from merge_asof misses) back to Python None so
    # callers get a uniform object-dtype column with None for unmatched rows.
    merged["sleep_stage"] = merged["sleep_stage"].where(
        merged["sleep_stage"].notna(), other=None
    )
    return merged


def expand_hypnogram(bedtime_start: pd.Timestamp, code: str) -> pd.DataFrame:
    """Expand an Oura hypnogram string into one row per 5-minute slot.

    Args:
        bedtime_start: local-naive Timestamp (Phase A invariant).
        code:          string of digits, one per 5-min slot.

    Returns:
        DataFrame with columns ``t`` (Timestamp) and ``stage`` (str | None).
        Empty DataFrame with the right columns if ``code`` is empty.
    """
    if not isinstance(code, str) or not code:
        return pd.DataFrame({"t": pd.Series(dtype="datetime64[ns]"),
                             "stage": pd.Series(dtype="object")})
    times = pd.date_range(bedtime_start, periods=len(code), freq=_FIVE_MIN)
    stages = [STAGE_CODES.get(c) for c in code]
    return pd.DataFrame({"t": times, "stage": stages})


_OUTPUT_COLUMNS = [
    "date",
    "session_glucose_deep_mean",
    "session_glucose_light_mean",
    "session_glucose_rem_mean",
    "session_glucose_awake_mean",
    "session_glucose_deep_minus_rem",
    "session_pct_time_high_during_deep",
]


def _assign_night(tagged: pd.DataFrame, sessions: pd.DataFrame) -> pd.Series:
    """Map each tagged CGM row to its sleep session's bedtime_start date.

    Uses merge_asof backward against bedtime_start so a reading at 02:00
    is assigned to the previous evening's session, not the wall-clock day.
    """
    if sessions.empty or tagged.empty:
        return pd.Series([pd.NaT] * len(tagged), index=tagged.index, dtype="datetime64[ns]")
    sess = sessions[["bedtime_start"]].sort_values("bedtime_start").reset_index(drop=True)
    sess["night"] = sess["bedtime_start"].dt.normalize()
    cgm = tagged.sort_values("timestamp").reset_index()
    merged = pd.merge_asof(
        cgm[["index", "timestamp"]],
        sess,
        left_on="timestamp",
        right_on="bedtime_start",
        direction="backward",
    )
    return merged.set_index("index")["night"].reindex(tagged.index)


def per_night_glucose_by_stage(
    tagged_cgm: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-night glucose-by-stage metrics.

    Args:
        tagged_cgm: CGM frame with ``timestamp``, ``glucose_mgdl``, ``sleep_stage``.
                    Rows where ``sleep_stage`` is NaN are dropped.
        sessions:   DataFrame with at least ``bedtime_start`` (local-naive).
                    Used to map each tagged reading to its session's night.

    Returns:
        DataFrame keyed by ``date`` (the calendar day of bedtime_start) with the
        six metric columns documented in the spec. Missing stages -> NaN.
    """
    empty_out = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    if tagged_cgm.empty:
        return empty_out

    df = tagged_cgm.dropna(subset=["sleep_stage"]).copy()
    if df.empty:
        return empty_out

    df["date"] = _assign_night(df, sessions)
    df = df.dropna(subset=["date"])
    if df.empty:
        return empty_out

    pivot = (
        df.groupby(["date", "sleep_stage"])["glucose_mgdl"]
          .mean()
          .unstack("sleep_stage")
    )
    for stage in ("deep", "light", "rem", "awake"):
        if stage not in pivot.columns:
            pivot[stage] = np.nan

    result = pd.DataFrame({
        "date": pivot.index,
        "session_glucose_deep_mean":  pivot["deep"].values,
        "session_glucose_light_mean": pivot["light"].values,
        "session_glucose_rem_mean":   pivot["rem"].values,
        "session_glucose_awake_mean": pivot["awake"].values,
    })
    result["session_glucose_deep_minus_rem"] = (
        result["session_glucose_deep_mean"] - result["session_glucose_rem_mean"]
    )

    deep_only = df[df["sleep_stage"] == "deep"]
    if deep_only.empty:
        result["session_pct_time_high_during_deep"] = np.nan
    else:
        deep_only = deep_only.assign(
            high=(deep_only["glucose_mgdl"] > cfg.GLUCOSE_HIGH).astype(float),
        )
        pct = deep_only.groupby("date")["high"].mean()
        result = result.merge(
            pct.rename("session_pct_time_high_during_deep").reset_index(),
            on="date", how="left",
        )

    return result[_OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)

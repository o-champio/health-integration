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

import pandas as pd

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

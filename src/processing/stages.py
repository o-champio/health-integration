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

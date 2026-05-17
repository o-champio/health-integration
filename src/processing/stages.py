"""Hypnogram expansion and per-night glucose-by-stage feature derivation.

Oura returns a ``sleep_phase_5_min`` string per sleep session — one digit
per 5-minute slot starting at ``bedtime_start``. Encoding:
1=deep, 2=light, 3=REM, 4=awake.

This module turns those strings into a time-indexed stage labels, tags a
CGM frame with the stage active at each reading, and computes the per-night
metrics consumed by the daily merge.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import settings as cfg

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

"""Tests for the activity / next-day TIR insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.activity_next_day import summary


def test_summary_uses_next_day_shift():
    """Activity on day X is paired with TIR on day X+1."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=4),
        "activity_high_activity_time": [60 * 30, 60 * 0, 60 * 45, 60 * 10],  # seconds
        "glucose_tir": [0.6, 0.7, 0.65, 0.55],
        "in_rest_mode": [False] * 4,
    })
    s = summary(df)
    # 4 rows -> 3 paired (last has no next-day TIR)
    assert s["caveat"]["n"] == 3
    # Activity is converted to minutes in the scatter
    assert (s["scatter"]["high_activity_min"] == [30, 0, 45]).all()


def test_summary_filters_rest_mode_both_sides():
    """Both the activity day AND its tir_next neighbor must be non-rest-mode."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=4),
        "activity_high_activity_time": [60 * 30, 60 * 0, 60 * 45, 60 * 10],
        "glucose_tir": [0.6, 0.7, 0.65, 0.55],
        "in_rest_mode": [False, True, False, False],
    })
    s = summary(df)
    # Row 0: dropped — its tir_next neighbor (row 1) is rest-mode.
    # Row 1: dropped — it IS rest-mode.
    # Row 2: kept — non-rest-mode and tir_next (row 3) is non-rest-mode.
    # Row 3: dropped — tir_next is NaN (no day 5).
    # Remaining valid pairs: row 2 only → n=1.
    assert s["caveat"]["n"] == 1

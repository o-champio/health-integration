"""Tests for the per-card summary() functions on the Insights hub."""
from __future__ import annotations

import pandas as pd


def test_activity_next_day_drops_rest_mode_neighbor():
    """If day X is in rest mode, drop day X-1 from the pair too — its tir_next
    column would otherwise be contaminated by a rest-mode day's glucose."""
    from app._insights.activity_next_day import summary

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "activity_high_activity_time": [60, 90, 120, 60, 90],
        "glucose_tir": [0.7, 0.8, 0.9, 0.6, 0.75],
        "in_rest_mode": [False, False, False, True, False],
    })
    s = summary(df)

    # Without the fix: row 2 pairs (activity=120 min) with tir_next=0.6 (a rest-mode day) → contamination.
    # With the fix: row 2 drops because its tir_next neighbor is rest-mode.
    # Row 3 drops because it IS rest-mode. Row 4 drops because tir_next is NaN.
    # Remaining valid pairs: row 0, row 1 → n=2.
    assert s["caveat"]["n"] == 2

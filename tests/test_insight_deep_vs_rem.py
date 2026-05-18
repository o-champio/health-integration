"""Tests for the deep-vs-REM glucose insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.deep_vs_rem import summary


def test_summary_basic_signal():
    """Three nights with deep < rem produce a negative mean delta and small p."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-05-01", periods=5),
        "session_glucose_deep_mean": [140.0, 130.0, 150.0, np.nan, 145.0],
        "session_glucose_rem_mean":  [160.0, 145.0, 170.0, 155.0, np.nan],
        "in_rest_mode": [False] * 5,
    })
    s = summary(df)
    # Nights with both deep+rem: 3 (indices 0,1,2)
    assert s["n"] == 3
    # All three have negative delta
    assert s["mean_delta"] < 0
    assert s["median_delta"] < 0
    assert 0.0 <= s["p"] <= 1.0


def test_summary_drops_rest_mode():
    df = pd.DataFrame({
        "date": pd.date_range("2026-05-01", periods=2),
        "session_glucose_deep_mean": [140.0, 1000.0],   # outlier on the rest day
        "session_glucose_rem_mean":  [160.0, 100.0],
        "in_rest_mode": [False, True],
    })
    s = summary(df)
    assert s["n"] == 1


def test_summary_empty_when_no_paired_nights():
    """If no row has both stages, return n=0 and NaN stats."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-05-01")],
        "session_glucose_deep_mean": [140.0],
        "session_glucose_rem_mean":  [np.nan],
        "in_rest_mode": [False],
    })
    s = summary(df)
    assert s["n"] == 0
    assert pd.isna(s["mean_delta"])
    assert pd.isna(s["median_delta"])
    assert pd.isna(s["p"])

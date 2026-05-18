"""Tests for the rolling TIR trend insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.rolling_tir import summary


def test_summary_window_means():
    """60 days of TIR rising from 0.5 to 0.8 — last window > first window."""
    dates = pd.date_range("2026-03-01", periods=60)
    tir = np.linspace(0.5, 0.8, 60)
    df = pd.DataFrame({"date": dates, "glucose_tir": tir,
                       "in_rest_mode": [False] * 60})
    s = summary(df)
    assert s["n"] == 60
    assert s["first_28d_mean"] < s["last_28d_mean"]
    assert abs(s["delta_pp"] - (s["last_28d_mean"] - s["first_28d_mean"]) * 100) < 1e-6


def test_summary_drops_rest_mode_from_rolling():
    """Rest-mode days are excluded before the rolling computation."""
    dates = pd.date_range("2026-03-01", periods=60)
    tir = np.full(60, 0.7)
    rest = [False] * 60
    rest[30] = True   # outlier day
    tir[30] = 0.1    # would tank the mean if not filtered
    df = pd.DataFrame({"date": dates, "glucose_tir": tir, "in_rest_mode": rest})
    s = summary(df)
    assert s["n"] == 59
    # Means should be ~0.7 — the outlier was filtered
    assert abs(s["first_28d_mean"] - 0.7) < 0.01
    assert abs(s["last_28d_mean"] - 0.7) < 0.01


def test_summary_insufficient_data():
    """Fewer than 28 valid days -> NaN window means but valid n."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=10),
        "glucose_tir": [0.7] * 10,
        "in_rest_mode": [False] * 10,
    })
    s = summary(df)
    assert s["n"] == 10
    assert pd.isna(s["first_28d_mean"])
    assert pd.isna(s["last_28d_mean"])
    assert pd.isna(s["delta_pp"])

"""Tests for the sleep-quality / next-day-glucose insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.sleep_next_day import summary


def test_summary_returns_caveat_shape():
    """Caveat dict has n, rho, p, monitored=True, significant."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=50),
        "sleep_score": rng.integers(60, 95, 50),
        "glucose_tir": rng.uniform(0.4, 0.9, 50),
        "in_rest_mode": [False] * 50,
    })
    s = summary(df)
    assert s["caveat"]["monitored"] is True
    assert "n" in s["caveat"]
    assert "rho" in s["caveat"]
    assert "p" in s["caveat"]
    assert "significant" in s["caveat"]
    assert s["caveat"]["n"] == 50
    # Scatter data exposed so render() can plot
    assert "scatter" in s
    assert {"sleep_score", "glucose_tir"}.issubset(s["scatter"].columns)


def test_summary_filters_rest_mode():
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=3),
        "sleep_score": [70, 80, 90],
        "glucose_tir": [0.5, 0.6, 0.7],
        "in_rest_mode": [False, True, False],
    })
    s = summary(df)
    assert s["caveat"]["n"] == 2


def test_summary_too_few_observations():
    """Below the minimum sample size, return n with NaN stats."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=3),
        "sleep_score": [70, 80, 90],
        "glucose_tir": [0.5, 0.6, 0.7],
        "in_rest_mode": [False] * 3,
    })
    s = summary(df)
    assert s["caveat"]["n"] == 3
    assert pd.isna(s["caveat"]["rho"])
    assert pd.isna(s["caveat"]["p"])

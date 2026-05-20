"""Tests for the overnight-HRV / glucose-CV insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.hrv_cv import summary


def test_summary_caveat_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=40),
        "session_avg_hrv": rng.normal(50, 10, 40),
        "glucose_cv": rng.uniform(20, 40, 40),
        "in_rest_mode": [False] * 40,
    })
    s = summary(df)
    assert s["caveat"]["monitored"] is True
    assert s["caveat"]["n"] == 40
    assert {"session_avg_hrv", "glucose_cv"}.issubset(s["scatter"].columns)


def test_summary_filters_rest_mode():
    df = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=3),
        "session_avg_hrv": [40, 50, 60],
        "glucose_cv": [30, 28, 25],
        "in_rest_mode": [False, True, False],
    })
    s = summary(df)
    assert s["caveat"]["n"] == 2

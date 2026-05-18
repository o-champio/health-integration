"""Tests for the hourly glucose pattern insight card."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights.hourly_pattern import summary


def test_summary_returns_24_hours():
    """Output has one row per hour, columns hour/median/q25/q75/n."""
    ts = pd.date_range("2026-05-01", periods=48, freq="30min")
    raw = pd.DataFrame({"timestamp": ts, "glucose_mgdl": np.arange(48) + 100})
    s = summary(raw)
    assert s["per_hour"].shape == (24, 5)
    assert list(s["per_hour"].columns) == ["hour", "median", "q25", "q75", "n"]


def test_summary_computes_dawn_rise():
    """Dawn rise = mean(05-08) - mean(02-05)."""
    rows = []
    for h, v in [(2, 90), (3, 90), (4, 90), (5, 110), (6, 120), (7, 110)]:
        rows.append({"timestamp": pd.Timestamp(f"2026-05-01 {h:02d}:00"),
                     "glucose_mgdl": v})
    raw = pd.DataFrame(rows)
    s = summary(raw)
    # mean(05-08) - mean(02-05) = mean(110,120,110) - mean(90,90,90) = 113.3 - 90 = 23.3
    assert abs(s["dawn_rise_mgdl"] - 23.333) < 0.1


def test_summary_empty_input():
    raw = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"),
                        "glucose_mgdl": pd.Series(dtype="float64")})
    s = summary(raw)
    assert s["per_hour"].empty
    assert pd.isna(s["dawn_rise_mgdl"])

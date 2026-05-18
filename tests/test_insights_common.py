"""Tests for app/_insights/_common.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app._insights._common import filter_rest_mode, monitored_caveat


def test_filter_rest_mode_drops_flagged_rows():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-02"),
                 pd.Timestamp("2026-05-03")],
        "x": [1.0, 2.0, 3.0],
        "in_rest_mode": [False, True, False],
    })
    out = filter_rest_mode(df)
    assert out["x"].tolist() == [1.0, 3.0]


def test_filter_rest_mode_handles_missing_column():
    """If the column is absent (legacy frames), pass through unchanged."""
    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = filter_rest_mode(df)
    assert out["x"].tolist() == [1.0, 2.0]


def test_filter_rest_mode_treats_nan_as_false():
    """Pre-v3 rows may have NaN — treat as 'not in rest mode' and keep them."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "in_rest_mode": [False, None, True]})
    out = filter_rest_mode(df)
    assert out["x"].tolist() == [1.0, 2.0]


def test_monitored_caveat_format():
    """Caveat dict has the fixed shape that monitored cards rely on."""
    cav = monitored_caveat(n=42, rho=0.13, p=0.14)
    assert cav == {"n": 42, "rho": 0.13, "p": 0.14, "monitored": True,
                   "significant": False}


def test_monitored_caveat_flags_significant_at_05():
    cav = monitored_caveat(n=100, rho=0.30, p=0.02)
    assert cav["significant"] is True

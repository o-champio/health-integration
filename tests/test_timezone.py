"""Tests for timezone normalization in Oura client and migrations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api.oura_client import _to_local_naive


# ── _to_local_naive ───────────────────────────────────────────────────────────

def test_to_local_naive_shifts_iso_with_offset():
    """An ISO timestamp in UTC should be shifted to local wall-clock and stripped."""
    s = pd.Series(["2025-03-14T23:30:00Z"])
    out = _to_local_naive(s)
    assert out.dt.tz is None
    assert out.iloc[0] == pd.Timestamp("2025-03-14 20:30:00")


def test_to_local_naive_handles_offset_string():
    """An ISO timestamp with non-UTC offset should also land on local wall-clock."""
    s = pd.Series(["2025-03-14T20:30:00-03:00"])  # already São Paulo local time
    out = _to_local_naive(s)
    assert out.iloc[0] == pd.Timestamp("2025-03-14 20:30:00")


def test_to_local_naive_preserves_nat():
    """Bad inputs should become NaT, not raise."""
    s = pd.Series(["not a date", None])
    out = _to_local_naive(s)
    assert out.isna().all()


def test_to_local_naive_empty_series():
    """Empty input should return an empty, tz-naive datetime series."""
    out = _to_local_naive(pd.Series([], dtype="object"))
    assert len(out) == 0
    assert out.dt.tz is None

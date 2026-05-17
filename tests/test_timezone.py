"""Tests for timezone normalization in Oura client and migrations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from unittest.mock import patch

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


# ── get_heartrate normalization ───────────────────────────────────────────────


def test_get_heartrate_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {"timestamp": "2025-03-14T23:30:00+00:00", "bpm": 60, "source": "awake"},
            {"timestamp": "2025-03-15T02:15:00+00:00", "bpm": 55, "source": "sleep"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_heartrate("2025-03-14T00:00:00", "2025-03-15T23:59:59")

    assert df["timestamp"].dt.tz is None
    # 23:30 UTC on Mar 14 = 20:30 local on Mar 14 (São Paulo, UTC-3)
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # 02:15 UTC on Mar 15 = 23:15 local on Mar 14
    assert df.iloc[1]["timestamp"] == pd.Timestamp("2025-03-14 23:15:00")

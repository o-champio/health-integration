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


# ── get_sleep_sessions normalization ─────────────────────────────────────────


def test_get_sleep_sessions_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {
                "id": "x",
                "day": "2025-03-14",
                "bedtime_start": "2025-03-15T02:00:00+00:00",  # 23:00 local Mar 14
                "bedtime_end":   "2025-03-15T10:00:00+00:00",  # 07:00 local Mar 15
                "average_hrv": 50,
            },
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_sleep_sessions("2025-03-14", "2025-03-15")

    assert df["bedtime_start"].dt.tz is None
    assert df.iloc[0]["bedtime_start"] == pd.Timestamp("2025-03-14 23:00:00")
    assert df.iloc[0]["bedtime_end"]   == pd.Timestamp("2025-03-15 07:00:00")


# ── get_workouts normalization ────────────────────────────────────────────────


def test_get_workouts_returns_local_naive():
    from src.api import oura_client

    fake = {
        "data": [
            {
                "id": "w",
                "day": "2025-03-14",
                "activity": "running",
                "start_datetime": "2025-03-14T22:00:00+00:00",  # 19:00 local
                "end_datetime":   "2025-03-14T23:00:00+00:00",  # 20:00 local
            },
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_workouts("2025-03-14", "2025-03-14")

    assert df["start_datetime"].dt.tz is None
    assert df.iloc[0]["start_datetime"] == pd.Timestamp("2025-03-14 19:00:00")
    assert df.iloc[0]["end_datetime"]   == pd.Timestamp("2025-03-14 20:00:00")


# ── End-to-end alignment ──────────────────────────────────────────────────────

def test_oura_event_near_utc_midnight_lands_on_correct_local_day():
    """An Oura HR sample at 23:30 local time on day X must land on day X locally,
    even though it is 02:30 UTC on day X+1."""
    from src.api import oura_client

    fake = {
        "data": [
            # 02:30 UTC on Mar 15 = 23:30 local on Mar 14
            {"timestamp": "2025-03-15T02:30:00+00:00", "bpm": 58, "source": "sleep"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_heartrate("2025-03-14", "2025-03-15")

    assert df.iloc[0]["timestamp"].date() == pd.Timestamp("2025-03-14").date()

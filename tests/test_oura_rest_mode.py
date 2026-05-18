"""Tests for Oura rest_mode_period endpoint client."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api import oura_client


def test_get_rest_mode_periods_empty():
    """No periods returned -> empty DataFrame with the documented columns."""
    with patch.object(oura_client, "_get", return_value={"data": []}):
        df = oura_client.get_rest_mode_periods("2026-04-01", "2026-04-30")
    assert df.empty
    assert list(df.columns) == ["start_date", "end_date"]


def test_get_rest_mode_periods_one_period():
    """A single period yields one row with local-naive Timestamp columns."""
    fake = {
        "data": [
            {"id": "r1", "start_day": "2026-04-15", "end_day": "2026-04-17"},
        ]
    }
    with patch.object(oura_client, "_get", return_value=fake):
        df = oura_client.get_rest_mode_periods("2026-04-01", "2026-04-30")

    assert len(df) == 1
    assert df["start_date"].dt.tz is None
    assert df.iloc[0]["start_date"] == pd.Timestamp("2026-04-15")
    assert df.iloc[0]["end_date"]   == pd.Timestamp("2026-04-17")

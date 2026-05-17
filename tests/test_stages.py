"""Tests for src/processing/stages.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.processing.stages import STAGE_CODES, expand_hypnogram


def test_expand_hypnogram_basic():
    """A 4-character hypnogram starting at 23:00 yields 4 rows at 5-min spacing."""
    bedtime_start = pd.Timestamp("2026-05-01 23:00:00")
    out = expand_hypnogram(bedtime_start, "1234")
    expected_times = [
        pd.Timestamp("2026-05-01 23:00:00"),
        pd.Timestamp("2026-05-01 23:05:00"),
        pd.Timestamp("2026-05-01 23:10:00"),
        pd.Timestamp("2026-05-01 23:15:00"),
    ]
    expected_stages = ["deep", "light", "rem", "awake"]
    assert out["t"].tolist() == expected_times
    assert out["stage"].tolist() == expected_stages


def test_expand_hypnogram_handles_unknown_codes():
    """Non-1234 characters become NaN stage labels but rows are still produced."""
    out = expand_hypnogram(pd.Timestamp("2026-05-01 23:00:00"), "1?2")
    assert out["stage"].tolist() == ["deep", None, "light"]


def test_expand_hypnogram_empty_string():
    """Empty hypnogram returns an empty DataFrame with the right columns."""
    out = expand_hypnogram(pd.Timestamp("2026-05-01 23:00:00"), "")
    assert out.empty
    assert list(out.columns) == ["t", "stage"]


def test_stage_codes_constant():
    """Oura encoding: 1=deep, 2=light, 3=REM, 4=awake."""
    assert STAGE_CODES == {"1": "deep", "2": "light", "3": "rem", "4": "awake"}

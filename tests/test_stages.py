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


from src.processing.stages import tag_cgm_with_stage


def test_tag_cgm_with_stage_inside_window():
    """CGM readings inside any sleep window get tagged; outside readings stay NaN."""
    sessions = pd.DataFrame({
        "bedtime_start": [pd.Timestamp("2026-05-01 23:00:00")],
        "sleep_phase_5_min": ["1122"],   # 23:00 deep, 23:05 deep, 23:10 light, 23:15 light
    })
    cgm = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-05-01 22:50:00"),   # before -> NaN
            pd.Timestamp("2026-05-01 23:01:00"),   # in slot 0 -> deep
            pd.Timestamp("2026-05-01 23:12:00"),   # in slot 2 -> light
            pd.Timestamp("2026-05-01 23:30:00"),   # after -> NaN
        ],
        "glucose_mgdl": [100.0, 110.0, 120.0, 130.0],
    })
    out = tag_cgm_with_stage(cgm, sessions)
    assert out["sleep_stage"].tolist() == [None, "deep", "light", None]


def test_tag_cgm_with_stage_tz_aware_input_raises():
    """Both inputs must be local-naive (Phase A invariant)."""
    sessions = pd.DataFrame({
        "bedtime_start": [pd.Timestamp("2026-05-01 23:00:00", tz="UTC")],
        "sleep_phase_5_min": ["12"],
    })
    cgm = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-05-01 23:00:00")],
        "glucose_mgdl": [100.0],
    })
    import pytest
    with pytest.raises(AssertionError):
        tag_cgm_with_stage(cgm, sessions)


def test_tag_cgm_with_stage_handles_empty_inputs():
    out_no_sessions = tag_cgm_with_stage(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-01")], "glucose_mgdl": [100.0]}),
        pd.DataFrame({"bedtime_start": pd.Series(dtype="datetime64[ns]"),
                      "sleep_phase_5_min": pd.Series(dtype="object")}),
    )
    assert out_no_sessions["sleep_stage"].isna().all()

    out_no_cgm = tag_cgm_with_stage(
        pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"),
                      "glucose_mgdl": pd.Series(dtype="float64")}),
        pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 23:00:00")],
                      "sleep_phase_5_min": ["12"]}),
    )
    assert out_no_cgm.empty
    assert "sleep_stage" in out_no_cgm.columns

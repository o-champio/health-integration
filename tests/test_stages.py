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


from src.processing.stages import per_night_glucose_by_stage


def test_per_night_glucose_by_stage_columns():
    """Two nights, all four stages present in both, verify the six expected columns."""
    tagged = pd.DataFrame({
        "timestamp": [
            # Night 1: 2026-05-01
            pd.Timestamp("2026-05-01 23:00:00"),  # deep, 100
            pd.Timestamp("2026-05-01 23:30:00"),  # rem, 120
            pd.Timestamp("2026-05-01 23:35:00"),  # light, 130
            pd.Timestamp("2026-05-01 23:40:00"),  # awake, 200
            # Night 2: 2026-05-02 (uses post-midnight readings)
            pd.Timestamp("2026-05-02 02:00:00"),  # deep, 180 (high)
            pd.Timestamp("2026-05-02 03:00:00"),  # rem, 150
        ],
        "glucose_mgdl": [100.0, 120.0, 130.0, 200.0, 180.0, 150.0],
        "sleep_stage": ["deep", "rem", "light", "awake", "deep", "rem"],
    })
    sessions = pd.DataFrame({
        "bedtime_start": [
            pd.Timestamp("2026-05-01 22:30:00"),
            pd.Timestamp("2026-05-02 01:00:00"),
        ],
    })
    out = per_night_glucose_by_stage(tagged, sessions)
    out = out.set_index("date")

    # Night 1 (date = 2026-05-01, the bedtime_start day)
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_deep_mean"]  == 100.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_rem_mean"]   == 120.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_light_mean"] == 130.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_awake_mean"] == 200.0
    assert out.loc[pd.Timestamp("2026-05-01"), "session_glucose_deep_minus_rem"] == -20.0
    # No high-glucose readings during deep on night 1
    assert out.loc[pd.Timestamp("2026-05-01"), "session_pct_time_high_during_deep"] == 0.0

    # Night 2 (date = 2026-05-02)
    assert out.loc[pd.Timestamp("2026-05-02"), "session_glucose_deep_mean"] == 180.0
    # 180 is NOT strictly > 180, so still 0.0 (use strict > cfg.GLUCOSE_HIGH)
    assert out.loc[pd.Timestamp("2026-05-02"), "session_pct_time_high_during_deep"] == 0.0


def test_per_night_glucose_by_stage_nan_for_missing_stage():
    """Stages absent from a night appear as NaN, not 0."""
    tagged = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-05-01 23:00:00")],
        "glucose_mgdl": [100.0],
        "sleep_stage": ["deep"],
    })
    sessions = pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 22:30:00")]})
    out = per_night_glucose_by_stage(tagged, sessions).set_index("date")
    row = out.loc[pd.Timestamp("2026-05-01")]
    assert row["session_glucose_deep_mean"] == 100.0
    assert pd.isna(row["session_glucose_rem_mean"])
    assert pd.isna(row["session_glucose_light_mean"])
    assert pd.isna(row["session_glucose_awake_mean"])
    assert pd.isna(row["session_glucose_deep_minus_rem"])  # rem missing -> diff NaN


def test_per_night_glucose_by_stage_high_threshold_uses_cfg():
    """Strict > cfg.GLUCOSE_HIGH (=180) — 181 counts, 180 does not."""
    tagged = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-05-01 23:00:00"),
            pd.Timestamp("2026-05-01 23:05:00"),
        ],
        "glucose_mgdl": [180.0, 181.0],
        "sleep_stage": ["deep", "deep"],
    })
    sessions = pd.DataFrame({"bedtime_start": [pd.Timestamp("2026-05-01 22:30:00")]})
    out = per_night_glucose_by_stage(tagged, sessions).set_index("date")
    # 1 of 2 readings is > 180 -> 0.5
    assert out.loc[pd.Timestamp("2026-05-01"), "session_pct_time_high_during_deep"] == 0.5


def test_per_night_glucose_by_stage_empty_input():
    """Empty tagged CGM -> empty output with the right columns."""
    tagged = pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "glucose_mgdl": pd.Series(dtype="float64"),
        "sleep_stage": pd.Series(dtype="object"),
    })
    out = per_night_glucose_by_stage(tagged, pd.DataFrame())
    assert out.empty
    expected = {
        "date",
        "session_glucose_deep_mean",
        "session_glucose_light_mean",
        "session_glucose_rem_mean",
        "session_glucose_awake_mean",
        "session_glucose_deep_minus_rem",
        "session_pct_time_high_during_deep",
    }
    assert set(out.columns) == expected

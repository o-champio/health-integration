"""Tests for parquet schema migrations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.processing.migrations import (
    migrate_v1_to_v2,
    read_schema_version,
    write_with_schema_version,
)
from config import settings as cfg


# ── migrate_v1_to_v2 ──────────────────────────────────────────────────────────

def test_migrate_v1_to_v2_daily_shifts_sleep_columns():
    """v1 stored sleep timestamps as UTC wall-clock; v2 stores them as local."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-03-14")],
        "bedtime_start": [pd.Timestamp("2025-03-15 02:00:00")],  # was 23:00 local
        "bedtime_end":   [pd.Timestamp("2025-03-15 10:00:00")],  # was 07:00 local
        "sleep_score":   [80],
    })
    out = migrate_v1_to_v2(df, kind="daily")
    assert out.iloc[0]["bedtime_start"] == pd.Timestamp("2025-03-14 23:00:00")
    assert out.iloc[0]["bedtime_end"]   == pd.Timestamp("2025-03-15 07:00:00")
    # Non-timestamp columns untouched.
    assert out.iloc[0]["sleep_score"] == 80


def test_migrate_v1_to_v2_highfreq_shifts_timestamp():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],  # was 20:30 local
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    out = migrate_v1_to_v2(df, kind="highfreq")
    assert out.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    assert out.iloc[0]["glucose_mgdl"] == 110.0


def test_migrate_v1_to_v2_is_NOT_idempotent_at_function_level():
    """Migration shifts every call — idempotency is enforced at the load path
    via the schema_version gate, NOT inside the function. This test documents
    that contract so callers don't bypass the gate."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-14 20:30:00")]})
    once = migrate_v1_to_v2(df, kind="highfreq")
    twice = migrate_v1_to_v2(once, kind="highfreq")
    # Two shifts of -3h each = -6h total. Intentional: never call this
    # function directly without checking schema_version first.
    assert twice.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 14:30:00")


def test_migrate_v1_to_v2_handles_missing_columns():
    """If a known column isn't present, migration is a no-op for that column."""
    df = pd.DataFrame({"date": [pd.Timestamp("2025-03-14")], "sleep_score": [80]})
    out = migrate_v1_to_v2(df, kind="daily")
    assert out.equals(df)


# ── schema_version metadata round-trip ────────────────────────────────────────

def test_write_and_read_schema_version(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "test.parquet"
    write_with_schema_version(df, path, version=2)
    assert read_schema_version(path) == 2


def test_read_schema_version_missing_metadata_returns_1(tmp_path):
    """Parquets written by older code have no metadata — treat as v1."""
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "legacy.parquet"
    df.to_parquet(path, index=False)  # no metadata
    assert read_schema_version(path) == 1


def test_read_schema_version_nonexistent_returns_1(tmp_path):
    path = tmp_path / "nope.parquet"
    assert read_schema_version(path) == 1


# ── pipeline load path integration ────────────────────────────────────────────

def test_pipeline_load_migrates_v1_parquet(tmp_path, monkeypatch):
    """Loading a v1-shaped parquet via _load_existing migrates and re-stamps it."""
    from src.processing import pipeline

    # Build a v1-style parquet: UTC-naive timestamps, no schema_version metadata.
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    path = tmp_path / "highfreq_merged.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    loaded = pipeline._load_existing(path)

    # Returned frame is migrated.
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # File on disk is rewritten with v2 stamp.
    assert read_schema_version(path) == cfg.SCHEMA_VERSION


def test_pipeline_load_skips_migration_when_current(tmp_path):
    from src.processing import pipeline

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-14 20:30:00")]})
    path = tmp_path / "highfreq_merged.parquet"
    write_with_schema_version(df, path, cfg.SCHEMA_VERSION)

    loaded = pipeline._load_existing(path)
    # No second shift was applied.
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")


def test_pipeline_load_restamps_non_oura_parquet(tmp_path):
    """A parquet with no Oura columns (e.g. glucose_readings) still gets the new stamp."""
    from src.processing import pipeline

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 10:00:00")],
        "glucose_mgdl": [110.0],
    })
    path = tmp_path / "glucose_readings.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    pipeline._load_existing(path)
    assert read_schema_version(path) == cfg.SCHEMA_VERSION


# ── v2 → v3 ───────────────────────────────────────────────────────────────────

from src.processing.migrations import migrate_v2_to_v3


def test_migrate_v2_to_v3_adds_daily_columns_with_defaults():
    """v2 daily parquet lacks the new columns; v3 adds them with NaN/False defaults."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-05-01")],
        "sleep_score": [80],
    })
    out = migrate_v2_to_v3(df, kind="daily")
    assert out.iloc[0]["sleep_score"] == 80
    assert pd.isna(out.iloc[0]["session_glucose_deep_mean"])
    assert pd.isna(out.iloc[0]["session_glucose_deep_minus_rem"])
    assert pd.isna(out.iloc[0]["session_pct_time_high_during_deep"])
    assert out.iloc[0]["in_rest_mode"] is False or out.iloc[0]["in_rest_mode"] == False  # noqa: E712


def test_migrate_v2_to_v3_adds_highfreq_sleep_stage_column():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-05-01 22:00:00")],
        "glucose_mgdl": [110.0],
    })
    out = migrate_v2_to_v3(df, kind="highfreq")
    assert "sleep_stage" in out.columns
    assert pd.isna(out.iloc[0]["sleep_stage"])


def test_migrate_v2_to_v3_does_not_overwrite_existing_columns():
    """If the v2 frame already has the columns (defensive), don't blank them out."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-05-01")],
        "session_glucose_deep_mean": [99.0],  # already present somehow
    })
    out = migrate_v2_to_v3(df, kind="daily")
    assert out.iloc[0]["session_glucose_deep_mean"] == 99.0


def test_migration_chains_v1_to_v3(tmp_path):
    """A v1-shaped parquet loaded via _load_existing migrates v1→v2→v3 in one go."""
    from src.processing import pipeline

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-03-14 23:30:00")],  # v1: UTC-naive
        "glucose_mgdl": [110.0],
        "bpm": [60],
    })
    path = tmp_path / "highfreq_merged.parquet"
    df.to_parquet(path, index=False)
    assert read_schema_version(path) == 1

    loaded = pipeline._load_existing(path)

    # v1→v2 shift applied
    assert loaded.iloc[0]["timestamp"] == pd.Timestamp("2025-03-14 20:30:00")
    # v2→v3 column added
    assert "sleep_stage" in loaded.columns
    # On-disk file is now v3
    assert read_schema_version(path) == cfg.SCHEMA_VERSION

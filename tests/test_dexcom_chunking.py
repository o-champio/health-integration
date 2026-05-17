"""Tests for Dexcom client's 30-day chunking behavior.

The Dexcom v3 /egvs and /events endpoints reject date ranges > 30 days with
HTTP 400. The client must split wide ranges into multiple requests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api.dexcom_client import _date_chunks


# ── _date_chunks ──────────────────────────────────────────────────────────────

def test_date_chunks_single_window_when_short():
    chunks = list(_date_chunks("2026-04-01", "2026-04-15"))
    assert chunks == [("2026-04-01", "2026-04-15")]


def test_date_chunks_splits_long_range():
    """A 50-day range must split into a 30-day chunk + a 20-day chunk."""
    chunks = list(_date_chunks("2026-03-28", "2026-05-17"))
    assert chunks == [
        ("2026-03-28", "2026-04-26"),
        ("2026-04-27", "2026-05-17"),
    ]


def test_date_chunks_exactly_30_days_is_single():
    chunks = list(_date_chunks("2026-04-01", "2026-04-30"))
    assert chunks == [("2026-04-01", "2026-04-30")]


def test_date_chunks_31_days_splits():
    chunks = list(_date_chunks("2026-04-01", "2026-05-01"))
    assert chunks == [
        ("2026-04-01", "2026-04-30"),
        ("2026-05-01", "2026-05-01"),
    ]


# ── get_egvs paginates across chunks ──────────────────────────────────────────

def test_get_egvs_makes_multiple_requests_for_long_range():
    """get_egvs over 50 days should issue 2 requests and concat the records."""
    from src.api import dexcom_client

    captured_params: list[dict] = []

    def fake_get(self, path, params=None):
        captured_params.append(params)
        if params["startDate"].startswith("2026-03-28"):
            return {"records": [
                {"systemTime": "2026-04-01T10:00:00+00:00", "value": 100,
                 "trend": "flat", "trendRate": 0.0},
            ]}
        return {"records": [
            {"systemTime": "2026-05-01T10:00:00+00:00", "value": 120,
             "trend": "flat", "trendRate": 0.0},
        ]}

    with patch.object(dexcom_client.DexcomClient, "_get", new=fake_get), \
         patch.object(dexcom_client.DexcomClient, "__init__", lambda self: None):
        client = dexcom_client.DexcomClient()
        df = client.get_egvs("2026-03-28", "2026-05-17")

    assert len(captured_params) == 2, f"expected 2 paginated calls, got {len(captured_params)}"
    assert captured_params[0]["startDate"] == "2026-03-28T00:00:00"
    assert captured_params[0]["endDate"]   == "2026-04-26T23:59:59"
    assert captured_params[1]["startDate"] == "2026-04-27T00:00:00"
    assert captured_params[1]["endDate"]   == "2026-05-17T23:59:59"
    assert len(df) == 2
    assert df["glucose_mg_dl"].tolist() == [100, 120]


def test_get_egvs_empty_response_still_returns_empty_df():
    from src.api import dexcom_client

    with patch.object(dexcom_client.DexcomClient, "_get", return_value={"records": []}), \
         patch.object(dexcom_client.DexcomClient, "__init__", lambda self: None):
        client = dexcom_client.DexcomClient()
        df = client.get_egvs("2026-04-01", "2026-04-15")

    assert df.empty
    assert list(df.columns) == ["timestamp", "glucose_mg_dl", "trend", "trend_rate"]

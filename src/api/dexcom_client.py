"""Dexcom API v3 client with automatic token refresh.

Reads the token from auth/tokens/dexcom_token.json (written by auth/oauth.py).
Run `python -m auth.oauth dexcom` once to authorize before using this client.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests

from config import settings as cfg

log = logging.getLogger(__name__)

_TOKEN_PATH = Path(cfg.DEXCOM_TOKEN_FILE)

# Dexcom v3 EGV and Events endpoints reject ranges > 30 days with HTTP 400.
_DEXCOM_MAX_DAYS = 30


def _date_chunks(start: str, end: str, max_days: int = _DEXCOM_MAX_DAYS) -> Iterator[tuple[str, str]]:
    """Yield (chunk_start, chunk_end) YYYY-MM-DD pairs in <= max_days windows."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(s + timedelta(days=max_days - 1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _load_token() -> dict:
    """Load token from disk, or fall back to DEXCOM_TOKEN_JSON env var.

    The env-var fallback is for hosted environments (e.g. Streamlit Cloud)
    where the local token file isn't present.
    """
    if _TOKEN_PATH.exists():
        with open(_TOKEN_PATH) as f:
            return json.load(f)

    env_blob = os.environ.get("DEXCOM_TOKEN_JSON")
    if env_blob:
        log.info("Loading Dexcom token from DEXCOM_TOKEN_JSON env var.")
        return json.loads(env_blob)

    raise FileNotFoundError(
        f"Dexcom token not found at {_TOKEN_PATH} and DEXCOM_TOKEN_JSON env "
        "var is unset. Run `python -m auth.oauth dexcom` to authorize."
    )


def _save_token(token: dict) -> None:
    """Persist refreshed token to disk. Silently no-ops on read-only filesystems."""
    try:
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKEN_PATH, "w") as f:
            json.dump(token, f)
    except OSError as exc:
        log.warning("Could not persist Dexcom token to %s: %s", _TOKEN_PATH, exc)


def _refresh_token(token: dict) -> dict:
    log.info("Refreshing Dexcom access token.")
    resp = requests.post(
        cfg.DEXCOM_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": token["refresh_token"],
            "redirect_uri": cfg.DEXCOM_REDIRECT_URI,
            "client_id": cfg.DEXCOM_CLIENT_ID,
            "client_secret": cfg.DEXCOM_CLIENT_SECRET,
        },
    )
    if resp.status_code != 200:
        log.error("Dexcom token refresh failed: %s", resp.text)
    resp.raise_for_status()
    new_token = resp.json()
    _save_token(new_token)
    return new_token


class DexcomClient:
    """Thin wrapper around the Dexcom API v3 with auto-refresh on 401."""

    def __init__(self) -> None:
        self._token = _load_token()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token['access_token']}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = cfg.DEXCOM_BASE_URL + path
        resp = requests.get(url, headers=self._headers(), params=params)
        if resp.status_code == 401:
            self._token = _refresh_token(self._token)
            resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Public methods ────────────────────────────────────────────────────────

    def _fetch_chunked(self, path: str, start_date: str, end_date: str) -> list[dict]:
        """Issue paginated requests in <= 30-day windows and concatenate ``records``."""
        all_records: list[dict] = []
        for chunk_start, chunk_end in _date_chunks(start_date, end_date):
            data = self._get(
                path,
                params={
                    "startDate": f"{chunk_start}T00:00:00",
                    "endDate":   f"{chunk_end}T23:59:59",
                },
            )
            all_records.extend(data.get("records", []))
        return all_records

    def get_egvs(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch estimated glucose values (EGVs) for a date range.

        Ranges longer than 30 days are split into multiple requests because the
        Dexcom v3 ``/egvs`` endpoint rejects wider windows with HTTP 400.

        Args:
            start_date: ISO date string, e.g. "2025-01-01"
            end_date:   ISO date string, e.g. "2025-03-14"

        Returns:
            DataFrame with columns: timestamp (local-naive), glucose_mg_dl, trend, trend_rate
        """
        records = self._fetch_chunked("users/self/egvs", start_date, end_date)

        if not records:
            log.warning("No EGV records returned for %s – %s", start_date, end_date)
            return pd.DataFrame(columns=["timestamp", "glucose_mg_dl", "trend", "trend_rate"])

        df = pd.DataFrame(records)
        df = df.rename(columns={"systemTime": "timestamp", "value": "glucose_mg_dl"})
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            .dt.tz_convert(cfg.LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )
        df = (
            df.drop_duplicates(subset=["timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

        keep = ["timestamp", "glucose_mg_dl", "trend", "trend_rate"]
        return df[[c for c in keep if c in df.columns]]

    def get_devices(self) -> list[dict]:
        """Return the list of Dexcom devices associated with the account."""
        return self._get("users/self/devices").get("devices", [])

    def get_events(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch user-logged events (carbs, insulin, exercise, health) for a date range.

        Dexcom v3 /users/self/events returns records with:
          - eventType: 'carbs' | 'insulin' | 'exercise' | 'health'
          - eventSubType: varies by type ('fastActing'/'longActing' for insulin, etc.)
          - value: amount (grams for carbs, units for insulin, minutes for exercise)
          - unit: unit of `value`
          - systemTime / displayTime: timestamps

        Returns:
            DataFrame with columns: timestamp, event_type, value, subtype
              event_type: 'insulin_rapid' | 'insulin_long' | 'food' | 'exercise'
              subtype: preserved Dexcom eventSubType (e.g. 'light'/'medium'/'heavy'
                       for exercise); NaN for food/insulin.
        """
        records = self._fetch_chunked("users/self/events", start_date, end_date)

        if not records:
            log.info("No Dexcom events returned for %s – %s", start_date, end_date)
            return pd.DataFrame(columns=["timestamp", "event_type", "value"])

        df = pd.DataFrame(records)

        ts_col = "systemTime" if "systemTime" in df.columns else "displayTime"
        df["timestamp"] = (
            pd.to_datetime(df[ts_col], format="ISO8601", utc=True)
            .dt.tz_convert(cfg.LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )

        def _map_type(row: pd.Series) -> str | None:
            etype = row.get("eventType")
            subtype = row.get("eventSubType") or ""
            if etype == "carbs":
                return "food"
            if etype == "insulin":
                return "insulin_long" if "long" in subtype.lower() else "insulin_rapid"
            if etype == "exercise":
                return "exercise"
            return None

        df["event_type"] = df.apply(_map_type, axis=1)
        df = df.dropna(subset=["event_type"])
        df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
        df["subtype"] = df.get("eventSubType")
        df.loc[df["event_type"] != "exercise", "subtype"] = np.nan

        return (
            df[["timestamp", "event_type", "value", "subtype"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

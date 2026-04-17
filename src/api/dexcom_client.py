"""Dexcom API v3 client with automatic token refresh.

Reads the token from auth/tokens/dexcom_token.json (written by auth/oauth.py).
Run `python -m auth.oauth dexcom` once to authorize before using this client.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import settings as cfg

log = logging.getLogger(__name__)

_TOKEN_PATH = Path(cfg.DEXCOM_TOKEN_FILE)


def _load_token() -> dict:
    if not _TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"Dexcom token not found at {_TOKEN_PATH}. "
            "Run `python -m auth.oauth dexcom` to authorize."
        )
    with open(_TOKEN_PATH) as f:
        return json.load(f)


def _save_token(token: dict) -> None:
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TOKEN_PATH, "w") as f:
        json.dump(token, f)


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

    def get_egvs(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch estimated glucose values (EGVs) for a date range.

        Args:
            start_date: ISO date string, e.g. "2025-01-01"
            end_date:   ISO date string, e.g. "2025-03-14"

        Returns:
            DataFrame with columns: timestamp (UTC, tz-naive), glucose_mg_dl, trend, trend_rate
        """
        # Dexcom v3 requires RFC 3339 datetimes
        start_dt = f"{start_date}T00:00:00"
        end_dt = f"{end_date}T23:59:59"

        data = self._get(
            "users/self/egvs",
            params={"startDate": start_dt, "endDate": end_dt},
        )

        records = data.get("records", [])
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
        df = df.sort_values("timestamp").reset_index(drop=True)

        keep = ["timestamp", "glucose_mg_dl", "trend", "trend_rate"]
        return df[[c for c in keep if c in df.columns]]

    def get_devices(self) -> list[dict]:
        """Return the list of Dexcom devices associated with the account."""
        return self._get("users/self/devices").get("devices", [])

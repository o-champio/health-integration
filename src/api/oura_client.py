"""Oura Ring API v2 client.

All public functions return tidy DataFrames ready for merging.
Requires a valid OAuth token -- run `python -m auth.oauth` first.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from config import settings as cfg

log = logging.getLogger(__name__)


# -- Timestamp normalization ---------------------------------------------------

def _to_local_naive(series: pd.Series) -> pd.Series:
    """Convert a series of ISO/UTC timestamps to local-naive in cfg.LOCAL_TIMEZONE.

    Oura returns timestamps with offsets (e.g. ``2025-03-14T23:30:00+00:00``).
    Stripping the offset directly leaves the wall-clock at UTC, which silently
    misaligns the data with CGM rows (already in local time). This helper does
    the conversion correctly: parse as UTC, convert to local, drop tz info.
    """
    return (
        pd.to_datetime(series, utc=True, errors="coerce")
          .dt.tz_convert(cfg.LOCAL_TIMEZONE)
          .dt.tz_localize(None)
    )


# -- Token / session -----------------------------------------------------------

def _load_token() -> dict:
    """Load the saved token from disk, or fall back to OURA_TOKEN_JSON env var.

    The env-var fallback is for hosted environments (e.g. Streamlit Cloud)
    where the local token file isn't present. On those platforms refreshed
    tokens are kept in memory for the container's lifetime; the env-var copy
    only re-seeds the next cold start.
    """
    path = Path(cfg.TOKEN_FILE)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    env_blob = os.environ.get("OURA_TOKEN_JSON")
    if env_blob:
        log.info("Loading Oura token from OURA_TOKEN_JSON env var.")
        return json.loads(env_blob)

    raise FileNotFoundError(
        f"No token at {cfg.TOKEN_FILE} and OURA_TOKEN_JSON env var is unset. "
        "Run `python -m auth.oauth` first."
    )


def _save_token(token: dict) -> None:
    """Persist the refreshed token to disk. Silently no-ops if the filesystem
    is read-only or ephemeral (e.g. Streamlit Cloud)."""
    try:
        Path(cfg.TOKEN_DIR).mkdir(parents=True, exist_ok=True)
        with open(cfg.TOKEN_FILE, "w") as f:
            json.dump(token, f)
    except OSError as exc:
        log.warning("Could not persist Oura token to %s: %s", cfg.TOKEN_FILE, exc)


# Module-level cache so refreshed tokens survive across calls when the
# filesystem is ephemeral (Streamlit Cloud) and _save_token is a no-op.
_TOKEN_CACHE: dict | None = None


def _get_token() -> dict:
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        _TOKEN_CACHE = _load_token()
    return _TOKEN_CACHE


def _set_token(token: dict) -> None:
    global _TOKEN_CACHE
    _TOKEN_CACHE = token
    _save_token(token)


def _refresh_token(token: dict) -> dict:
    """Exchange the refresh_token for a fresh access_token."""
    log.info("Refreshing Oura access token...")
    resp = requests.post(
        cfg.TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": token["refresh_token"],
            "client_id": cfg.OURA_CLIENT_ID,
            "client_secret": cfg.OURA_CLIENT_SECRET,
        },
    )
    if not resp.ok:
        log.error("Token refresh failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()
    new_token = resp.json()
    _set_token(new_token)
    log.info("Token refreshed and saved.")
    return new_token


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Authenticated GET to Oura v2 API. Auto-refreshes on 401."""
    token = _get_token()
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    url = f"{cfg.BASE_URL}{endpoint}"

    resp = requests.get(url, headers=headers, params=params or {})

    if resp.status_code == 401:
        log.warning("Got 401; attempting token refresh.")
        token = _refresh_token(token)
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        resp = requests.get(url, headers=headers, params=params or {})

    if not resp.ok:
        log.error("Oura API error %s %s: %s", resp.status_code, endpoint, resp.text)
    resp.raise_for_status()
    return resp.json()


# -- Personal ------------------------------------------------------------------

def get_personal_info() -> dict:
    """Return raw user profile dict."""
    return _get("personal_info")


# -- Daily summaries (one row per calendar day) --------------------------------

def _daily(endpoint: str, start_date: str, end_date: str, prefix: str) -> pd.DataFrame:
    """Generic daily-summary fetcher. Prefixes columns and renames 'day' to 'date'."""
    raw = _get(endpoint, {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return df
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["id"], errors="ignore")
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "date"})
    return df.sort_values("date").reset_index(drop=True)


def get_daily_sleep(start_date: str, end_date: str) -> pd.DataFrame:
    return _daily("daily_sleep", start_date, end_date, "sleep")


def get_daily_readiness(start_date: str, end_date: str) -> pd.DataFrame:
    return _daily("daily_readiness", start_date, end_date, "readiness")


def get_daily_activity(start_date: str, end_date: str) -> pd.DataFrame:
    return _daily("daily_activity", start_date, end_date, "activity")


def get_daily_stress(start_date: str, end_date: str) -> pd.DataFrame:
    return _daily("daily_stress", start_date, end_date, "stress")


def get_daily_spo2(start_date: str, end_date: str) -> pd.DataFrame:
    return _daily("daily_spo2", start_date, end_date, "spo2")


# -- High-frequency data -------------------------------------------------------

def get_heartrate(start_datetime: str, end_datetime: str) -> pd.DataFrame:
    """5-minute-resolution heart rate readings.

    Args:
        start_datetime: ISO-8601 e.g. '2024-01-01T00:00:00'
        end_datetime:   ISO-8601 e.g. '2024-01-07T23:59:59'
    """
    raw = _get("heartrate", {"start_datetime": start_datetime, "end_datetime": end_datetime})
    df = pd.DataFrame(raw.get("data", []))
    if df.empty:
        return df
    df["timestamp"] = _to_local_naive(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def get_sleep_sessions(start_date: str, end_date: str) -> pd.DataFrame:
    """Individual sleep sessions with detailed hypnogram-level metrics."""
    raw = _get("sleep", {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return df
    df = df.drop(columns=["id"], errors="ignore")
    for col in ["bedtime_start", "bedtime_end"]:
        if col in df.columns:
            df[col] = _to_local_naive(df[col])
    if "day" in df.columns:
        # Oura's `day` is a local-date string (e.g. "2025-03-14") — parse as naive date.
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df.sort_values("day").reset_index(drop=True)


def get_workouts(start_date: str, end_date: str) -> pd.DataFrame:
    """Workout sessions: type, duration, calories, HR zone distribution."""
    raw = _get("workout", {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return df
    df = df.drop(columns=["id"], errors="ignore")
    df["day"] = pd.to_datetime(df.get("day"), errors="coerce")
    for col in ["start_datetime", "end_datetime"]:
        if col in df.columns:
            df[col] = _to_local_naive(df[col])
    return df.sort_values("day").reset_index(drop=True)


def get_rest_mode_periods(start_date: str, end_date: str) -> pd.DataFrame:
    """Periods where Oura was in rest mode (illness, recovery).

    Returns a DataFrame with columns ``start_date``, ``end_date`` (local-naive
    Timestamps). Used by Phase C analyses as an outlier filter — days
    overlapping any rest-mode period should typically be excluded from
    correlations.
    """
    raw = _get("rest_mode_period", {"start_date": start_date, "end_date": end_date})
    df = pd.json_normalize(raw.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["start_date", "end_date"])
    df = df.drop(columns=["id"], errors="ignore")
    df["start_date"] = pd.to_datetime(df.get("start_day"), errors="coerce")
    df["end_date"]   = pd.to_datetime(df.get("end_day"), errors="coerce")
    return df[["start_date", "end_date"]].sort_values("start_date").reset_index(drop=True)

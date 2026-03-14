"""Health data pipeline: merge Oura Ring + LibreLink CGM data.

Public API
----------
build_daily_dataset(start_date, end_date)
    -> unified daily DataFrame (glucose stats + all Oura daily summaries)

build_highfreq_dataset(start_date, end_date)
    -> high-frequency DataFrame: CGM readings + Oura heart-rate joined on timestamp

load_glucose_only()
    -> (glucose_readings, daily_stats) without any Oura calls
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import settings as cfg
from src.api import libre_client, oura_client

log = logging.getLogger(__name__)

# Oura API rejects queries spanning more than ~30 days
_OURA_MAX_DAYS = 30


# -- LibreLink helpers ---------------------------------------------------------

def load_glucose_only(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LibreLink data without touching the Oura API.

    Returns (glucose_readings, daily_stats).
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR
    raw = libre_client.load_all(data_dir)
    glucose = libre_client.get_glucose_readings(raw)
    daily = libre_client.daily_glucose_stats(glucose)
    log.info(
        "Loaded %d glucose readings across %d days.",
        len(glucose), len(daily),
    )
    return glucose, daily


# -- Oura helpers --------------------------------------------------------------

def _date_chunks(start: str, end: str, max_days: int = _OURA_MAX_DAYS):
    """Yield (chunk_start, chunk_end) strings in max_days-wide windows."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(s + timedelta(days=max_days - 1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _fetch_oura_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all Oura daily metrics, chunking large date ranges."""
    fetchers = [
        oura_client.get_daily_sleep,
        oura_client.get_daily_readiness,
        oura_client.get_daily_activity,
        oura_client.get_daily_stress,
    ]

    all_frames: list[pd.DataFrame] = []
    for fn in fetchers:
        chunks = []
        for cs, ce in _date_chunks(start_date, end_date):
            try:
                df = fn(cs, ce)
                if not df.empty:
                    chunks.append(df)
            except Exception as exc:
                log.warning("%s (%s..%s): %s", fn.__name__, cs, ce, exc)
        if chunks:
            all_frames.append(pd.concat(chunks, ignore_index=True))

    if not all_frames:
        return pd.DataFrame()

    merged = all_frames[0]
    for df in all_frames[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    return merged.sort_values("date").reset_index(drop=True)


# -- Feature engineering -------------------------------------------------------

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-night lag columns for sleep and readiness scores.

    Effects of poor sleep / low HRV are typically observed the following day.
    """
    lag_cols = [c for c in df.columns if c.startswith(("sleep_score", "readiness_score"))]
    if not lag_cols:
        return df
    df = df.sort_values("date").copy()
    for col in lag_cols:
        df[f"{col}_prev_night"] = df[col].shift(1)
    log.debug("Added %d lag features.", len(lag_cols))
    return df


def _add_glucose_variability(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling glucose variability features over a 7-day window."""
    if "glucose_mean" not in df.columns or len(df) < 3:
        return df
    df = df.sort_values("date").copy()
    df["glucose_mean_7d"] = df["glucose_mean"].rolling(7, min_periods=3).mean().round(2)
    df["glucose_std_7d"] = df["glucose_mean"].rolling(7, min_periods=3).std().round(2)
    return df


# -- Persistence (incremental save/load) ---------------------------------------

def _load_existing(path: Path) -> pd.DataFrame | None:
    """Load an existing processed parquet file, if it exists."""
    if path.exists():
        df = pd.read_parquet(path)
        log.info("Loaded existing processed data: %d rows from %s", len(df), path.name)
        return df
    return None


def _save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save a processed DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Saved %d rows to %s", len(df), path.name)


# -- Public API ----------------------------------------------------------------

DAILY_PARQUET = cfg.DATA_PROCESSED_DIR / "daily_merged.parquet"
HIGHFREQ_PARQUET = cfg.DATA_PROCESSED_DIR / "highfreq_merged.parquet"


def build_daily_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str | Path | None = None,
    incremental: bool = True,
) -> pd.DataFrame:
    """Build the main daily dataset.

    Rows: one per calendar day.
    Columns: glucose stats (always) + Oura daily metrics (if token exists)
             + lag features + rolling variability.

    When incremental=True, loads existing processed data and only fetches
    new dates from the Oura API.
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR
    glucose, daily_glucose = load_glucose_only(data_dir)

    if start_date is None:
        start_date = daily_glucose["date"].min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = daily_glucose["date"].max().strftime("%Y-%m-%d")

    daily_glucose = daily_glucose[
        (daily_glucose["date"] >= pd.Timestamp(start_date))
        & (daily_glucose["date"] <= pd.Timestamp(end_date))
    ].copy()

    # Determine effective start_date for Oura fetch (incremental)
    oura_start = start_date
    existing = _load_existing(DAILY_PARQUET) if incremental else None
    if existing is not None and "date" in existing.columns:
        last_date = pd.Timestamp(existing["date"].max())
        candidate = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if candidate <= end_date:
            oura_start = candidate
            log.info("Incremental mode: fetching Oura data from %s", oura_start)
        else:
            log.info("Processed data is up to date through %s.", last_date.strftime("%Y-%m-%d"))

    try:
        oura_df = _fetch_oura_daily(oura_start, end_date)
    except Exception as exc:
        log.error("Oura API unavailable (%s). Returning glucose-only dataset.", exc)
        return daily_glucose.sort_values("date").reset_index(drop=True)

    if oura_df.empty and existing is None:
        result = daily_glucose.sort_values("date").reset_index(drop=True)
    elif oura_df.empty:
        merged = pd.merge(daily_glucose, existing, on="date", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)
        result = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = pd.merge(daily_glucose, oura_df, on="date", how="left")
        if existing is not None:
            # Combine: keep existing rows for dates not in the new fetch
            oura_cols = [c for c in oura_df.columns if c != "date"]
            existing_only = existing[
                existing["date"] < pd.Timestamp(oura_start)
            ]
            if not existing_only.empty:
                merged = pd.concat([existing_only, merged], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date"], keep="last")
        result = merged.sort_values("date").reset_index(drop=True)

    result = _add_lag_features(result)
    result = _add_glucose_variability(result)

    _save_processed(result, DAILY_PARQUET)
    return result


def build_highfreq_dataset(
    start_date: str,
    end_date: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build a high-frequency dataset: CGM readings + Oura heart rate.

    CGM (~15 min) and Oura HR (~5 min) are joined with an asof merge
    (nearest Oura HR within +/-10 min of each glucose reading).
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR
    raw = libre_client.load_all(data_dir)
    glucose = libre_client.get_glucose_readings(raw)

    glucose = glucose[
        (glucose["timestamp"] >= pd.Timestamp(start_date))
        & (glucose["timestamp"] <= pd.Timestamp(end_date))
    ].copy()

    try:
        hr_chunks = []
        for cs, ce in _date_chunks(start_date, end_date):
            hr = oura_client.get_heartrate(f"{cs}T00:00:00", f"{ce}T23:59:59")
            if not hr.empty:
                hr_chunks.append(hr)
        if not hr_chunks:
            log.warning("No heart-rate data returned. Returning glucose only.")
            return glucose
        hr = pd.concat(hr_chunks, ignore_index=True)
    except Exception as exc:
        log.error("Heart-rate fetch failed (%s). Returning glucose only.", exc)
        return glucose

    glucose = glucose.sort_values("timestamp")
    hr = hr.sort_values("timestamp")

    merged = pd.merge_asof(
        glucose,
        hr[["timestamp", "bpm"]],
        on="timestamp",
        tolerance=pd.Timedelta("10min"),
        direction="nearest",
    )
    result = merged.reset_index(drop=True)

    _save_processed(result, HIGHFREQ_PARQUET)
    return result

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

import contextlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import settings as cfg
from src.api import libre_client, oura_client

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _timed(label: str):
    """Log elapsed time for a pipeline stage at INFO level with a [PERF] prefix."""
    t0 = time.perf_counter()
    yield
    log.info("[PERF] %-45s %.3fs", label, time.perf_counter() - t0)

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
    with _timed("CSV load (load_all)"):
        raw = libre_client.load_all(data_dir)
    glucose = libre_client.get_glucose_readings(raw)
    with _timed("CSV stats (daily_glucose_stats)"):
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


def _fetch_sleep_sessions(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch detailed sleep sessions and aggregate to one row per day.

    Extracts actual physiological values (average_hrv in ms, deep_sleep_duration
    in seconds, lowest_heart_rate in bpm) that the daily summaries do not provide.
    When multiple sessions exist for one day, keeps the longest (primary) session.
    """
    chunks = []
    for cs, ce in _date_chunks(start_date, end_date):
        try:
            with _timed(f"Oura sleep_sessions {cs}..{ce}"):
                df = oura_client.get_sleep_sessions(cs, ce)
            if not df.empty:
                chunks.append(df)
        except Exception as exc:
            log.warning("get_sleep_sessions (%s..%s): %s", cs, ce, exc)
    if not chunks:
        return pd.DataFrame()

    sessions = pd.concat(chunks, ignore_index=True)

    # Keep only long_sleep (primary nightly sessions)
    if "type" in sessions.columns:
        long = sessions[sessions["type"] == "long_sleep"]
        if not long.empty:
            sessions = long

    # Pick the longest session per day when duplicates exist
    if "total_sleep_duration" in sessions.columns:
        sessions = (
            sessions.sort_values("total_sleep_duration", ascending=False)
            .drop_duplicates(subset=["day"], keep="first")
        )

    keep_cols = ["day"]
    rename_map = {"day": "date"}
    physio_cols = {
        "average_hrv": "session_avg_hrv",
        "average_heart_rate": "session_avg_hr",
        "lowest_heart_rate": "session_lowest_hr",
        "deep_sleep_duration": "session_deep_sleep_sec",
        "rem_sleep_duration": "session_rem_sleep_sec",
        "total_sleep_duration": "session_total_sleep_sec",
        "efficiency": "session_efficiency",
        "restless_periods": "session_restless_periods",
    }
    for src, dst in physio_cols.items():
        if src in sessions.columns:
            keep_cols.append(src)
            rename_map[src] = dst

    result = sessions[keep_cols].rename(columns=rename_map).copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()

    # Convert durations from seconds to minutes for readability
    for col in ["session_deep_sleep_sec", "session_rem_sleep_sec", "session_total_sleep_sec"]:
        min_col = col.replace("_sec", "_min")
        if col in result.columns:
            result[min_col] = (result[col] / 60).round(1)
            result = result.drop(columns=[col])

    return result.sort_values("date").reset_index(drop=True)


def _fetch_oura_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all Oura daily metrics + sleep sessions, chunking large date ranges."""
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
                with _timed(f"Oura {fn.__name__} {cs}..{ce}"):
                    df = fn(cs, ce)
                if not df.empty:
                    chunks.append(df)
            except Exception as exc:
                log.warning("%s (%s..%s): %s", fn.__name__, cs, ce, exc)
        if chunks:
            all_frames.append(pd.concat(chunks, ignore_index=True))

    # Also fetch detailed sleep sessions for physiological values
    sessions_df = _fetch_sleep_sessions(start_date, end_date)
    if not sessions_df.empty:
        all_frames.append(sessions_df)

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
        with _timed(f"Load existing parquet ({path.name})"):
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

    # Determine effective start_date for Oura fetch (incremental).
    # Never fetch before OURA_START_DATE — ring wasn't worn before that.
    oura_start = max(start_date, cfg.OURA_START_DATE)
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
        with _timed("Oura full fetch"):
            oura_df = _fetch_oura_daily(oura_start, end_date)
    except Exception as exc:
        log.error("Oura API unavailable (%s). Using cached Oura data.", exc)
        oura_df = pd.DataFrame()

    with _timed("Merge glucose + Oura"):
        # Step 1: merge fresh glucose with new Oura data (may be empty)
        if oura_df.empty:
            result = daily_glucose.copy()
        else:
            result = pd.merge(daily_glucose, oura_df, on="date", how="left")

        # Step 2: backfill Oura columns from existing parquet where the
        # new merge has gaps (old dates, or API failure).  Glucose columns
        # are always taken from the fresh CSV and never overwritten.
        if existing is not None:
            glucose_col_set = set(daily_glucose.columns)
            existing_indexed = existing.set_index("date")
            for col in existing_indexed.columns:
                if col in glucose_col_set:
                    continue
                if col not in result.columns:
                    result[col] = pd.NA
                mask = result[col].isna()
                if mask.any():
                    result.loc[mask, col] = (
                        result.loc[mask, "date"].map(existing_indexed[col])
                    )

        result = result.sort_values("date").reset_index(drop=True)

    with _timed("Lag + rolling features"):
        result = _add_lag_features(result)
        result = _add_glucose_variability(result)

    with _timed("Save parquet"):
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

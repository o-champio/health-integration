"""Health data pipeline: merge Oura Ring + CGM data with incremental caching.

All datasets are persisted as parquet files in data/processed/. On each run,
only new data (since the last saved timestamp/date) is fetched from APIs,
then appended to the existing parquet. This minimizes API calls and speeds
up dashboard loads.

Public API
----------
sync_all()
    -> run all pipelines incrementally (glucose, daily, workouts, high-freq)

build_daily_dataset(start_date, end_date)
    -> unified daily DataFrame (glucose stats + all Oura daily summaries)

build_highfreq_dataset(start_date, end_date)
    -> high-frequency DataFrame: CGM readings + Oura heart-rate joined on timestamp

load_glucose_only()
    -> (glucose_readings, daily_stats) without any Oura calls

fetch_workouts(start_date, end_date)
    -> workout sessions from Oura
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


# -- Persistence ---------------------------------------------------------------

def _load_existing(path: Path) -> pd.DataFrame | None:
    """Load an existing processed parquet file, if it exists."""
    if path.exists():
        with _timed(f"Load parquet ({path.name})"):
            df = pd.read_parquet(path)
        log.info("Loaded %d rows from %s", len(df), path.name)
        return df
    return None


def _save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save a processed DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Saved %d rows to %s", len(df), path.name)


def _append_and_dedupe(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    sort_col: str,
    dedupe_col: str | None = None,
) -> pd.DataFrame:
    """Append new rows to existing DataFrame, deduplicate, and sort."""
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    if dedupe_col and dedupe_col in combined.columns:
        combined = combined.drop_duplicates(subset=[dedupe_col], keep="last")
    elif dedupe_col is None:
        combined = combined.drop_duplicates(keep="last")
    return combined.sort_values(sort_col).reset_index(drop=True)


# -- Parquet paths -------------------------------------------------------------

GLUCOSE_PARQUET  = cfg.GLUCOSE_PARQUET
DAILY_PARQUET    = cfg.DATA_PROCESSED_DIR / "daily_merged.parquet"
HIGHFREQ_PARQUET = cfg.DATA_PROCESSED_DIR / "highfreq_merged.parquet"
WORKOUT_PARQUET  = cfg.DATA_PROCESSED_DIR / "workouts.parquet"


# -- Date helpers --------------------------------------------------------------

def _date_chunks(start: str, end: str, max_days: int = _OURA_MAX_DAYS):
    """Yield (chunk_start, chunk_end) strings in max_days-wide windows."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(s + timedelta(days=max_days - 1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _today() -> str:
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


# -- Glucose (incremental) ----------------------------------------------------

def _load_dexcom_glucose(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Dexcom EGVs and normalize to match LibreLink schema."""
    try:
        from src.api.dexcom_client import DexcomClient
        df = DexcomClient().get_egvs(start_date, end_date)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "glucose_mgdl"])
        df = df.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
        return df[["timestamp", "glucose_mgdl"]].dropna(subset=["glucose_mgdl"]).reset_index(drop=True)
    except FileNotFoundError:
        log.info("Dexcom token not found — skipping Dexcom fetch.")
        return pd.DataFrame(columns=["timestamp", "glucose_mgdl"])
    except Exception as exc:
        log.warning("Dexcom fetch failed (%s). Skipping.", exc)
        return pd.DataFrame(columns=["timestamp", "glucose_mgdl"])


def _load_libre_glucose(data_dir: str | Path, end_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load LibreLink CSV glucose readings, optionally only after end_ts."""
    try:
        with _timed("CSV load (load_all)"):
            raw = libre_client.load_all(data_dir)
        g = libre_client.get_glucose_readings(raw)
        cutover = pd.Timestamp(cfg.CUTOVER_DATE)
        g = g[g["timestamp"] < cutover]
        if end_ts is not None:
            g = g[g["timestamp"] > end_ts]
        return g
    except FileNotFoundError:
        log.info("No LibreLink CSVs found — skipping.")
        return pd.DataFrame(columns=["timestamp", "glucose_mgdl"])


def sync_glucose(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Incrementally sync glucose readings to GLUCOSE_PARQUET.

    Reads existing parquet, determines what's missing, fetches only new data
    from LibreLink CSVs and/or Dexcom API, appends, saves, and returns all.
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR

    existing = _load_existing(GLUCOSE_PARQUET)
    cutover = pd.Timestamp(cfg.CUTOVER_DATE)
    today = pd.Timestamp.today().normalize()

    last_ts = None
    if existing is not None and not existing.empty:
        last_ts = existing["timestamp"].max()
        log.info("Glucose cache up to %s", last_ts)

    frames: list[pd.DataFrame] = []

    # LibreLink: reload CSVs for new pre-cutover readings
    if last_ts is None or last_ts < cutover:
        libre_g = _load_libre_glucose(data_dir, end_ts=last_ts)
        if not libre_g.empty:
            frames.append(libre_g)
            log.info("LibreLink: %d new readings.", len(libre_g))

    # Dexcom: fetch post-cutover data from where we left off
    if today >= cutover:
        dex_start = cutover
        if last_ts is not None and last_ts >= cutover:
            dex_start = (last_ts + pd.Timedelta(minutes=1))
        dex_start_str = dex_start.strftime("%Y-%m-%d")
        dex_end_str = today.strftime("%Y-%m-%d")
        if dex_start_str <= dex_end_str:
            with _timed("Dexcom EGV fetch (incremental)"):
                dex_g = _load_dexcom_glucose(dex_start_str, dex_end_str)
            if not dex_g.empty:
                # Filter out readings we already have (overlap from date-level query)
                if last_ts is not None:
                    dex_g = dex_g[dex_g["timestamp"] > last_ts]
                if not dex_g.empty:
                    frames.append(dex_g)
                    log.info("Dexcom: %d new readings.", len(dex_g))

    if not frames and existing is not None and not existing.empty:
        log.info("Glucose data is up to date.")
        return existing

    new_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "glucose_mgdl"])
    result = _append_and_dedupe(existing, new_data, sort_col="timestamp", dedupe_col="timestamp")

    if result.empty:
        raise RuntimeError(
            "No glucose data available. "
            "Add LibreLink CSVs to data/raw/ or run `python -m auth.oauth dexcom`."
        )

    _save_processed(result, GLUCOSE_PARQUET)
    return result


DAILY_GLUCOSE_STATS_PARQUET = cfg.DATA_PROCESSED_DIR / "daily_glucose_stats.parquet"


def load_glucose_only(
    data_dir: str | Path | None = None,
    _glucose: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load glucose data incrementally (parquet-cached).

    Pass _glucose to skip the sync step (used by sync_all to avoid redundant calls).
    Returns (glucose_readings, daily_stats).
    """
    glucose = _glucose if _glucose is not None else sync_glucose(data_dir)

    # Check if daily stats cache is still valid
    existing_stats = _load_existing(DAILY_GLUCOSE_STATS_PARQUET)
    if existing_stats is not None and not existing_stats.empty:
        cached_max = existing_stats["date"].max()
        glucose_max = glucose["timestamp"].max().normalize()
        if cached_max >= glucose_max:
            log.info("Daily glucose stats cache is up to date (%d days).", len(existing_stats))
            return glucose, existing_stats

    with _timed("Daily glucose stats (recompute)"):
        daily = libre_client.daily_glucose_stats(glucose)
    _save_processed(daily, DAILY_GLUCOSE_STATS_PARQUET)
    log.info("Loaded %d glucose readings across %d days.", len(glucose), len(daily))
    return glucose, daily


# -- Workouts (incremental) ---------------------------------------------------

def fetch_workouts(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Incrementally sync Oura workouts to WORKOUT_PARQUET."""
    if start_date is None:
        start_date = cfg.OURA_START_DATE
    if end_date is None:
        end_date = _today()

    existing = _load_existing(WORKOUT_PARQUET)

    # Determine fetch window: start from day after last cached workout
    fetch_start = start_date
    if existing is not None and not existing.empty and "day" in existing.columns:
        last_day = pd.Timestamp(existing["day"].max())
        candidate = (last_day + timedelta(days=1)).strftime("%Y-%m-%d")
        if candidate > end_date:
            log.info("Workout data is up to date through %s.", last_day.strftime("%Y-%m-%d"))
            return existing
        fetch_start = candidate
        log.info("Workouts: fetching from %s (cached through %s)", fetch_start, last_day.strftime("%Y-%m-%d"))

    chunks = []
    for cs, ce in _date_chunks(fetch_start, end_date):
        try:
            with _timed(f"Oura workouts {cs}..{ce}"):
                df = oura_client.get_workouts(cs, ce)
            if not df.empty:
                chunks.append(df)
        except Exception as exc:
            log.warning("get_workouts (%s..%s): %s", cs, ce, exc)

    if not chunks and existing is not None:
        _save_processed(existing, WORKOUT_PARQUET)
        return existing

    new_data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    result = _append_and_dedupe(existing, new_data, sort_col="day")
    _save_processed(result, WORKOUT_PARQUET)
    return result


# -- Oura daily helpers --------------------------------------------------------

def _fetch_sleep_sessions(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch detailed sleep sessions and aggregate to one row per day."""
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

    if "type" in sessions.columns:
        long = sessions[sessions["type"] == "long_sleep"]
        if not long.empty:
            sessions = long

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
    lag_cols = [c for c in df.columns if c.startswith(("sleep_score", "readiness_score"))]
    if not lag_cols:
        return df
    df = df.sort_values("date").copy()
    for col in lag_cols:
        df[f"{col}_prev_night"] = df[col].shift(1)
    return df


def _add_glucose_variability(df: pd.DataFrame) -> pd.DataFrame:
    if "glucose_mean" not in df.columns or len(df) < 3:
        return df
    df = df.sort_values("date").copy()
    df["glucose_mean_7d"] = df["glucose_mean"].rolling(7, min_periods=3).mean().round(2)
    df["glucose_std_7d"] = df["glucose_mean"].rolling(7, min_periods=3).std().round(2)
    return df


# -- Daily dataset (incremental) -----------------------------------------------

def build_daily_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str | Path | None = None,
    incremental: bool = True,
    _glucose: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the main daily dataset (incremental).

    Rows: one per calendar day.
    Columns: glucose stats + Oura daily metrics + lag features + rolling variability.

    Pass _glucose to reuse an already-synced glucose DataFrame.
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR
    glucose, daily_glucose = load_glucose_only(data_dir, _glucose=_glucose)

    if start_date is None:
        start_date = daily_glucose["date"].min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = max(daily_glucose["date"].max().strftime("%Y-%m-%d"), _today())

    daily_glucose = daily_glucose[
        (daily_glucose["date"] >= pd.Timestamp(start_date))
        & (daily_glucose["date"] <= pd.Timestamp(end_date))
    ].copy()

    # Determine effective Oura fetch start (incremental).
    # Always re-fetch the last 2 days because Oura data for "today" updates
    # throughout the day (sleep finalizes in the morning, activity during the day).
    oura_start = max(start_date, cfg.OURA_START_DATE)
    existing = _load_existing(DAILY_PARQUET) if incremental else None
    if existing is not None and "date" in existing.columns:
        last_date = pd.Timestamp(existing["date"].max())
        # Re-fetch from 2 days before last_date to catch intra-day Oura updates
        refetch_from = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
        oura_start = max(oura_start, refetch_from)
        log.info("Incremental mode: fetching Oura data from %s (last cached: %s)",
                 oura_start, last_date.strftime("%Y-%m-%d"))

    try:
        with _timed("Oura fetch (%s .. %s)" % (oura_start, end_date)):
            oura_df = _fetch_oura_daily(oura_start, end_date)
    except Exception as exc:
        log.error("Oura API unavailable (%s). Using cached Oura data.", exc)
        oura_df = pd.DataFrame()

    with _timed("Merge glucose + Oura"):
        if oura_df.empty:
            result = daily_glucose.copy()
        else:
            result = pd.merge(daily_glucose, oura_df, on="date", how="left")

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


# -- High-freq dataset (incremental) ------------------------------------------

def build_highfreq_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str | Path | None = None,
    _glucose: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a high-frequency dataset: CGM readings + Oura heart rate.

    Incrementally appends to HIGHFREQ_PARQUET — only fetches data for dates
    not yet in the cached parquet.
    Pass _glucose to reuse an already-synced glucose DataFrame.
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR

    existing = _load_existing(HIGHFREQ_PARQUET)

    # Determine fetch window
    if existing is not None and not existing.empty:
        last_ts = existing["timestamp"].max()
        fetch_start = (last_ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")
        if start_date is None or fetch_start > start_date:
            start_date = fetch_start
        log.info("High-freq: cached through %s, fetching from %s", last_ts, start_date)

    if start_date is None:
        start_date = cfg.OURA_START_DATE
    if end_date is None:
        end_date = _today()

    if start_date > end_date:
        log.info("High-freq data is up to date.")
        return existing

    # Fetch new glucose and HR for the missing window
    glucose = _glucose if _glucose is not None else sync_glucose(data_dir)
    glucose = glucose[
        (glucose["timestamp"] >= pd.Timestamp(start_date))
        & (glucose["timestamp"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
    ]

    if glucose.empty:
        log.info("No new glucose data for high-freq.")
        return existing if existing is not None else pd.DataFrame()

    try:
        hr_chunks = []
        for cs, ce in _date_chunks(start_date, end_date):
            hr = oura_client.get_heartrate(f"{cs}T00:00:00", f"{ce}T23:59:59")
            if not hr.empty:
                hr_chunks.append(hr)
        if not hr_chunks:
            log.warning("No heart-rate data returned for new window.")
            new_merged = glucose.copy()
        else:
            hr = pd.concat(hr_chunks, ignore_index=True).sort_values("timestamp")
            glucose = glucose.sort_values("timestamp")
            new_merged = pd.merge_asof(
                glucose,
                hr[["timestamp", "bpm"]],
                on="timestamp",
                tolerance=pd.Timedelta("10min"),
                direction="nearest",
            )
    except Exception as exc:
        log.error("Heart-rate fetch failed (%s).", exc)
        new_merged = glucose.copy()

    result = _append_and_dedupe(existing, new_merged, sort_col="timestamp", dedupe_col="timestamp")
    _save_processed(result, HIGHFREQ_PARQUET)
    return result


# -- Unified sync --------------------------------------------------------------

def sync_all(
    data_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run all pipelines incrementally. Returns dict of all datasets.

    This is the single entry point that ensures all parquets are up to date.
    Each pipeline reads its parquet, fetches only missing data from APIs,
    appends, and saves.
    """
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR

    results = {}

    with _timed("sync_all: glucose"):
        glucose = sync_glucose(data_dir)
        results["glucose"] = glucose

    with _timed("sync_all: daily"):
        results["daily"] = build_daily_dataset(data_dir=data_dir, _glucose=glucose)

    with _timed("sync_all: workouts"):
        results["workouts"] = fetch_workouts()

    with _timed("sync_all: highfreq"):
        results["highfreq"] = build_highfreq_dataset(data_dir=data_dir, _glucose=glucose)

    log.info(
        "sync_all complete: glucose=%d, daily=%d, workouts=%d, highfreq=%d",
        len(results["glucose"]), len(results["daily"]),
        len(results["workouts"]), len(results["highfreq"]),
    )
    return results

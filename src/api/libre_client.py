"""LibreLink CGM data loading and processing.

Handles FreeStyle LibreLink / Libre 3 export CSVs.
The export format has one metadata row before the column headers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings as cfg

log = logging.getLogger(__name__)

# Record type codes in the LibreLink export
RECORD_TYPES = {
    0: "historic_glucose",   # Automatic CGM reading (~15-min interval)
    1: "scan_glucose",       # Manual NFC scan
    2: "strip_glucose",      # Blood-glucose strip
    3: "insulin_rapid",      # Rapid-acting insulin dose
    4: "food",               # Food / carb entry
    5: "insulin_long",       # Long-acting insulin dose
    6: "device_event",       # Sensor start/stop or other device event
}


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a single LibreLink export CSV.

    The file has one metadata header line (row 0) followed by column headers
    on row 1, so we skip row 0.  Timestamps are localtime with no timezone.
    """
    filepath = Path(filepath)
    log.debug("Loading CSV: %s", filepath.name)
    df = pd.read_csv(filepath, skiprows=1, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Device Timestamp"] = pd.to_datetime(
        df["Device Timestamp"], format="%m-%d-%Y %I:%M %p", errors="coerce"
    )
    df = df.dropna(subset=["Device Timestamp"])
    df = df.sort_values("Device Timestamp").reset_index(drop=True)
    return df


def load_all(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and concatenate all CSVs in data_dir, dropping exact duplicates."""
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir!r}")
    log.info("Loading %d CSV file(s) from %s", len(files), data_dir)
    frames = [load_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["Device Timestamp", "Record Type"]).sort_values(
        "Device Timestamp"
    ).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d duplicate rows.", dropped)
    return df


# -- CGM readings --------------------------------------------------------------

def get_glucose_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Extract CGM readings (Record Type 0) as a clean time-series.

    Returns columns: timestamp, glucose_mgdl
    """
    mask = df["Record Type"] == 0
    g = df.loc[mask, ["Device Timestamp", "Historic Glucose mg/dL"]].copy()
    g.columns = ["timestamp", "glucose_mgdl"]
    g["glucose_mgdl"] = pd.to_numeric(g["glucose_mgdl"], errors="coerce")
    g = g.dropna(subset=["glucose_mgdl"]).reset_index(drop=True)
    return g


def get_scan_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Extract manual scan readings (Record Type 1)."""
    mask = df["Record Type"] == 1
    g = df.loc[mask, ["Device Timestamp", "Scan Glucose mg/dL"]].copy()
    g.columns = ["timestamp", "glucose_mgdl"]
    g["glucose_mgdl"] = pd.to_numeric(g["glucose_mgdl"], errors="coerce")
    return g.dropna(subset=["glucose_mgdl"]).reset_index(drop=True)


# -- Daily aggregations --------------------------------------------------------

def daily_glucose_stats(glucose: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day glucose statistics from CGM readings.

    Columns: date, glucose_mean, glucose_std, glucose_min, glucose_max,
    glucose_readings, glucose_tir, glucose_tbr, glucose_tar, glucose_cv, glucose_gmi
    """
    g = glucose.copy()
    g["date"] = g["timestamp"].dt.normalize()

    # Single groupby pass: compute all stats at once using a custom aggregator
    # to avoid 5 separate groupby scans over the same data.
    low, high = cfg.GLUCOSE_LOW, cfg.GLUCOSE_HIGH

    def _agg(vals: pd.Series) -> pd.Series:
        n = len(vals)
        mean = vals.mean()
        std = vals.std()
        in_r = ((vals >= low) & (vals <= high)).sum()
        below = (vals < low).sum()
        above = (vals > high).sum()
        return pd.Series({
            "glucose_mean": round(mean, 2),
            "glucose_std": round(std, 2),
            "glucose_min": round(vals.min(), 2),
            "glucose_max": round(vals.max(), 2),
            "glucose_readings": n,
            "glucose_tir": round(in_r / n, 3),
            "glucose_tbr": round(below / n, 3),
            "glucose_tar": round(above / n, 3),
            "glucose_cv": round(std / mean, 3) if mean else float("nan"),
            "glucose_gmi": round(3.31 + 0.02392 * mean, 2),
        })

    stats = g.groupby("date")["glucose_mgdl"].apply(_agg)
    return stats.reset_index()


# -- Convenience ---------------------------------------------------------------

def glucose_date_range(glucose: pd.DataFrame) -> tuple[str, str]:
    """Return (start_date, end_date) as 'YYYY-MM-DD' strings."""
    start = glucose["timestamp"].min().strftime("%Y-%m-%d")
    end = glucose["timestamp"].max().strftime("%Y-%m-%d")
    return start, end

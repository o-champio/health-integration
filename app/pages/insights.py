"""Insights hub — landing page composing the six curated cards.

Cards A-C are descriptive / validated; cards D-F are literature-monitored
correlations with explicit uncertainty caveats. Rest-mode days are dropped
inside each card's ``summary()``.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app._insights import (
    activity_next_day,
    deep_vs_rem,
    hourly_pattern,
    hrv_cv,
    rolling_tir,
    sleep_next_day,
)


def _latest(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    sub = df.dropna(subset=[col])
    if sub.empty:
        return None
    val = sub.iloc[-1][col]
    return float(val) if pd.notna(val) else None


def _snapshot(df: pd.DataFrame) -> None:
    """Five 'latest available value' metric cards at the top of the hub."""
    tir = _latest(df, "glucose_tir")
    mean_g = _latest(df, "glucose_mean")
    sleep = _latest(df, "prev_night_sleep_score")
    readiness = _latest(df, "prev_day_readiness_score")
    activity = _latest(df, "prev_day_activity_score")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Time in Range", f"{tir:.1%}" if tir is not None else "—")
    c2.metric("Mean Glucose", f"{mean_g:.0f} mg/dL" if mean_g is not None else "—")
    c3.metric("Sleep Score", f"{sleep:.0f}" if sleep is not None else "—")
    c4.metric("Readiness", f"{readiness:.0f}" if readiness is not None else "—")
    c5.metric("Activity Score", f"{activity:.0f}" if activity is not None else "—")


def render(df: pd.DataFrame, raw_glucose: pd.DataFrame) -> None:
    """Render the full insights hub.

    Args:
        df:           daily-merged frame (one row per date).
        raw_glucose:  high-frequency CGM frame for hourly pattern.
    """
    st.title("Insights")
    st.caption(
        "Curated findings. Validated cards show effects we measured in your data. "
        "Monitored cards report relationships from published research; we keep "
        "tracking whether they show in your data as N grows."
    )

    _snapshot(df)
    st.divider()

    # Row 1: validated / descriptive
    c1, c2 = st.columns(2)
    with c1:
        deep_vs_rem.render(df)
    with c2:
        hourly_pattern.render(raw_glucose)

    rolling_tir.render(df)

    c3, c4 = st.columns(2)
    with c3:
        sleep_next_day.render(df)
    with c4:
        hrv_cv.render(df)

    activity_next_day.render(df)

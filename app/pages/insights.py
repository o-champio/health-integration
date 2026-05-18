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

    # Row 1: validated / descriptive
    c1, c2 = st.columns(2)
    with c1:
        deep_vs_rem.render(df)
    with c2:
        hourly_pattern.render(raw_glucose)

    rolling_tir.render(df)

    st.divider()
    st.subheader("Literature-monitored relationships")
    st.caption(
        "These cards reflect published associations. Our N is currently small; "
        "we report Spearman rho with explicit uncertainty so you don't over-interpret."
    )

    c3, c4 = st.columns(2)
    with c3:
        sleep_next_day.render(df)
    with c4:
        hrv_cv.render(df)

    activity_next_day.render(df)

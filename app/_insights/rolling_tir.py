"""Insight C: Rolling TIR trend (descriptive).

28-day rolling time-in-range with a first-vs-last-window delta. Pure
descriptive — answers "am I trending up or down".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app._insights._common import filter_rest_mode


def summary(df: pd.DataFrame) -> dict:
    """Return rolling-TIR series and first/last 28-day window means."""
    df = filter_rest_mode(df)
    df = df.dropna(subset=["glucose_tir"]).sort_values("date")
    n = len(df)
    if n < 28:
        return {
            "n": n,
            "rolling": pd.DataFrame({"date": df["date"], "tir_28d": [np.nan] * n}),
            "first_28d_mean": np.nan,
            "last_28d_mean":  np.nan,
            "delta_pp":       np.nan,
        }
    rolling = df["glucose_tir"].rolling(28, min_periods=10).mean()
    rolling_df = pd.DataFrame({"date": df["date"].values, "tir_28d": rolling.values})
    valid = rolling.dropna()
    first = valid.iloc[:28].mean()
    last  = valid.iloc[-28:].mean()
    return {
        "n": n,
        "rolling": rolling_df,
        "first_28d_mean": float(first),
        "last_28d_mean":  float(last),
        "delta_pp":       float((last - first) * 100.0),
    }


def render(df: pd.DataFrame) -> None:
    s = summary(df)
    st.subheader("Rolling TIR trend")
    if s["n"] < 28:
        st.info(f"Need at least 28 days of data; have {s['n']}.")
        return
    direction = "up" if s["delta_pp"] >= 0 else "down"
    st.markdown(
        f"28-day rolling time-in-range has moved **{direction} "
        f"{abs(s['delta_pp']):.1f} percentage points** from the first window "
        f"({s['first_28d_mean']*100:.0f}%) to the most recent ({s['last_28d_mean']*100:.0f}%)."
    )
    r = s["rolling"]
    fig = go.Figure(go.Scatter(x=r["date"], y=r["tir_28d"] * 100, mode="lines"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="28-day rolling TIR (%)",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

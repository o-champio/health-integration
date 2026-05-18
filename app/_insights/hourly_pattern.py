"""Insight B: Hourly glucose pattern (descriptive).

Median glucose by hour-of-day with IQR band. Surfaces dawn rise and
post-meal patterns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def summary(raw: pd.DataFrame) -> dict:
    """Return median/IQR glucose per hour and a dawn-rise scalar.

    ``raw`` is the high-frequency CGM frame with ``timestamp`` and
    ``glucose_mgdl`` columns. No rest-mode filter (descriptive only).
    """
    if raw.empty:
        empty = pd.DataFrame(columns=["hour", "median", "q25", "q75", "n"])
        return {"per_hour": empty, "dawn_rise_mgdl": np.nan}
    df = raw.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    per_hour = (
        df.groupby("hour")["glucose_mgdl"]
          .agg(median="median",
               q25=lambda s: s.quantile(0.25),
               q75=lambda s: s.quantile(0.75),
               n="count")
          .reset_index()
    )
    # Ensure all 24 hours appear (NaN for hours with no data)
    full = pd.DataFrame({"hour": range(24)}).merge(per_hour, on="hour", how="left")

    morning = full[full["hour"].between(5, 7)]["median"].mean()
    night   = full[full["hour"].between(2, 4)]["median"].mean()
    dawn_rise = morning - night if not (pd.isna(morning) or pd.isna(night)) else np.nan
    return {"per_hour": full, "dawn_rise_mgdl": float(dawn_rise) if not pd.isna(dawn_rise) else np.nan}


def render(raw: pd.DataFrame) -> None:
    s = summary(raw)
    st.subheader("Hourly glucose pattern")
    if s["per_hour"].empty:
        st.info("No CGM data available.")
        return
    if not pd.isna(s["dawn_rise_mgdl"]):
        st.markdown(
            f"Dawn rise (05-07 vs 02-04 medians): **{s['dawn_rise_mgdl']:+.1f} mg/dL**."
        )
    ph = s["per_hour"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph["hour"], y=ph["q75"], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ph["hour"], y=ph["q25"], mode="lines", fill="tonexty",
                             line=dict(width=0), showlegend=False, name="IQR"))
    fig.add_trace(go.Scatter(x=ph["hour"], y=ph["median"], mode="lines+markers",
                             name="Median"))
    fig.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="Glucose (mg/dL)",
        xaxis=dict(tickmode="linear", dtick=2),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

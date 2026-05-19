"""Insight A: Deep sleep vs REM glucose.

Validated in Phase B (n=121, deep - rem = -13.2 mg/dL, p < 0.0001).
This is the headline insight; biology: growth hormone spikes during deep
sleep, counter-regulatory hormones quiet, so glucose dips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as st_

from app._insights._common import filter_rest_mode
from app._shared import chart


def summary(df: pd.DataFrame) -> dict:
    """Return per-night delta statistics for deep vs REM glucose.

    Drops rest-mode rows and rows where either stage mean is NaN.
    """
    df = filter_rest_mode(df)
    pair = df[["session_glucose_deep_mean", "session_glucose_rem_mean"]].dropna()
    n = len(pair)
    if n == 0:
        return {"n": 0, "mean_delta": np.nan, "median_delta": np.nan, "p": np.nan}
    deltas = pair["session_glucose_deep_mean"] - pair["session_glucose_rem_mean"]
    # Wilcoxon test for paired diff; one-sided H1: deep < rem
    # scipy handles n >= 1; returns valid p even for small samples
    try:
        _, p = st_.wilcoxon(deltas, alternative="less")
    except ValueError:
        # e.g. all deltas are zero (wilcoxon undefined)
        p = float("nan")
    return {
        "n": int(n),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(deltas.median()),
        "p": float(p) if not np.isnan(p) else np.nan,
    }


def render(df: pd.DataFrame) -> None:
    """Render the deep-vs-REM insight card."""
    s = summary(df)
    st.subheader("Deep sleep & nighttime glucose")
    if s["n"] == 0:
        st.info("Not enough paired-stage nights yet to compute this.")
        return
    delta = s["mean_delta"]
    direction = "lower" if delta < 0 else "higher"
    sig = "" if pd.isna(s["p"]) else f"  (Wilcoxon p={s['p']:.4f})"
    st.markdown(
        f"Across **{s['n']} nights**, your glucose runs **{abs(delta):.1f} mg/dL "
        f"{direction}** in deep sleep than in REM.{sig}"
    )

    df_f = filter_rest_mode(df)
    pair = df_f[["session_glucose_deep_mean", "session_glucose_rem_mean"]].dropna()
    deltas = pair["session_glucose_deep_mean"] - pair["session_glucose_rem_mean"]
    fig = go.Figure(go.Histogram(x=deltas, nbinsx=30))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(
        xaxis_title="Deep mean − REM mean (mg/dL)",
        yaxis_title="Nights",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    chart(fig)

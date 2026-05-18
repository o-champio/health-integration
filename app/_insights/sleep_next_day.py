"""Insight D: Sleep quality and same-day glucose (literature-monitored).

Oura's sleep `day` already equals the wake-up day, so `sleep_score` on day X
already represents the night before X. Correlate sleep_score against the
SAME day's TIR (no lag needed).

Effect size in our data is currently small (~rho=0.09, n=135), but the
relationship is well-published; we show the scatter with a caveat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as st_

from app._insights._common import filter_rest_mode, monitored_caveat

_MIN_N = 10


def summary(df: pd.DataFrame) -> dict:
    df = filter_rest_mode(df)
    pair = df[["sleep_score", "glucose_tir"]].dropna()
    n = len(pair)
    if n < _MIN_N:
        cav = {"n": int(n), "rho": np.nan, "p": np.nan, "monitored": True,
               "significant": False}
    else:
        rho, p = st_.spearmanr(pair["sleep_score"], pair["glucose_tir"])
        cav = monitored_caveat(n=n, rho=rho, p=p)
    return {"caveat": cav, "scatter": pair.reset_index(drop=True)}


def render(df: pd.DataFrame) -> None:
    s = summary(df)
    cav = s["caveat"]
    st.subheader("Sleep quality & next-day glucose")
    if cav["n"] < _MIN_N:
        st.info(f"Need at least {_MIN_N} days with both metrics; have {cav['n']}.")
        return
    if cav["significant"]:
        copy = (f"With **n={cav['n']}** days, Spearman ρ = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Signal is present.")
    else:
        copy = (f"With **n={cav['n']}** days, Spearman ρ = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Not yet statistically significant — based on "
                f"published associations between sleep quality and glycemic control, "
                f"we keep tracking.")
    st.markdown(copy)
    scatter = s["scatter"]
    fig = go.Figure(go.Scatter(x=scatter["sleep_score"], y=scatter["glucose_tir"] * 100,
                               mode="markers"))
    fig.update_layout(
        xaxis_title="Sleep score",
        yaxis_title="Glucose TIR (%)",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

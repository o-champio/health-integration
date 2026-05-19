"""Insight F: Activity load and next-day TIR (literature-monitored).

High-activity minutes on day X are paired with TIR on day X+1. Our data:
rho=+0.13, p=0.14 across n=131. Direction matches literature
(more exercise -> better next-day glycemic control); not yet significant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as st_

from app._insights._common import filter_rest_mode, monitored_caveat
from app._shared import chart

_MIN_N = 10


def summary(df: pd.DataFrame) -> dict:
    df = filter_rest_mode(df).sort_values("date").copy()
    if "activity_high_activity_time" not in df.columns:
        cav = {"n": 0, "rho": np.nan, "p": np.nan, "monitored": True,
               "significant": False}
        empty = pd.DataFrame(columns=["high_activity_min", "tir_next"])
        return {"caveat": cav, "scatter": empty}

    df["high_activity_min"] = df["activity_high_activity_time"] / 60.0
    df["tir_next"] = df["glucose_tir"].shift(-1)
    pair = df[["high_activity_min", "tir_next"]].dropna()
    n = len(pair)
    if n < _MIN_N:
        cav = {"n": int(n), "rho": np.nan, "p": np.nan, "monitored": True,
               "significant": False}
    else:
        rho, p = st_.spearmanr(pair["high_activity_min"], pair["tir_next"])
        cav = monitored_caveat(n=n, rho=rho, p=p)
    return {"caveat": cav, "scatter": pair.reset_index(drop=True)}


def render(df: pd.DataFrame) -> None:
    s = summary(df)
    cav = s["caveat"]
    st.subheader("Activity & next-day TIR")
    if cav["n"] < _MIN_N:
        st.info(f"Need at least {_MIN_N} paired days; have {cav['n']}.")
        return
    if cav["significant"]:
        copy = (f"With **n={cav['n']}** pairs, Spearman rho = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Signal is present.")
    else:
        copy = (f"With **n={cav['n']}** pairs, Spearman rho = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Not yet statistically significant — exercise "
                f"is associated with improved next-day glycemic control in published "
                f"work, so we keep tracking.")
    st.markdown(copy)
    sc = s["scatter"]
    fig = go.Figure(go.Scatter(x=sc["high_activity_min"], y=sc["tir_next"] * 100,
                               mode="markers"))
    fig.update_layout(
        xaxis_title="High-activity minutes (day X)",
        yaxis_title="Next-day TIR (%)",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    chart(fig)

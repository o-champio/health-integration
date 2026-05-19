"""Insight E: Overnight HRV and glucose variability (literature-monitored).

Higher HRV reflects parasympathetic dominance and is associated with
better glycemic control in healthy adults. Our data: rho=-0.11, p=0.21
across n=129; not yet significant but the direction matches the literature.
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
    df = filter_rest_mode(df)
    pair = df[["session_avg_hrv", "glucose_cv"]].dropna()
    n = len(pair)
    if n < _MIN_N:
        cav = {"n": int(n), "rho": np.nan, "p": np.nan, "monitored": True,
               "significant": False}
    else:
        rho, p = st_.spearmanr(pair["session_avg_hrv"], pair["glucose_cv"])
        cav = monitored_caveat(n=n, rho=rho, p=p)
    return {"caveat": cav, "scatter": pair.reset_index(drop=True)}


def render(df: pd.DataFrame) -> None:
    s = summary(df)
    cav = s["caveat"]
    st.subheader("HRV & glucose variability")
    if cav["n"] < _MIN_N:
        st.info(f"Need at least {_MIN_N} days with both metrics; have {cav['n']}.")
        return
    if cav["significant"]:
        copy = (f"With **n={cav['n']}** nights, Spearman rho = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Signal is present.")
    else:
        copy = (f"With **n={cav['n']}** nights, Spearman rho = **{cav['rho']:+.2f}** "
                f"(p={cav['p']:.3f}). Not yet statistically significant — higher "
                f"overnight HRV is associated with lower glycemic variability in "
                f"published studies, so we keep tracking.")
    st.markdown(copy)
    sc = s["scatter"]
    fig = go.Figure(go.Scatter(x=sc["session_avg_hrv"], y=sc["glucose_cv"],
                               mode="markers"))
    fig.update_layout(
        xaxis_title="Overnight average HRV (ms)",
        yaxis_title="Glucose CV (%)",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    chart(fig)

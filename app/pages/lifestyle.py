"""Lifestyle Factors — sleep architecture and activity/readiness.

Phase C drop list:
- HRV & Stress tab (the HRV insight lives on the hub).

Phase B integration: the Sleep tab now shows per-night glucose-by-stage
from session_glucose_{deep,light,rem,awake}_mean columns.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app._shared import _avail, _dual_axis_chart, _label, chart
from app._theme import C


def render(df: pd.DataFrame) -> None:
    tab_sleep, tab_activity = st.tabs(["Sleep architecture", "Activity & Readiness"])
    with tab_sleep:
        _sleep(df)
    with tab_activity:
        _activity(df)


def _sleep(df: pd.DataFrame) -> None:
    """Sleep architecture trends + Phase B per-night glucose-by-stage.

    Body copied from app/main.py:_lifestyle_sleep, then extended with the
    per-night glucose-by-stage section below.
    """
    st.markdown("#### Sleep vs Glucose Control")
    smooth = st.session_state.get("smooth_window", 7)
    sleep_cols = _avail(df, "Sleep")
    glucose_cols = _avail(df, "Glucose")
    if not sleep_cols:
        st.info("No sleep data available.")
        return

    c1, c2 = st.columns(2)
    y_sleep = c1.selectbox("Sleep metric", sleep_cols, format_func=_label, key="sl_y1")
    y_gluc = c2.selectbox("Glucose metric", glucose_cols, format_func=_label, key="sl_y2") if glucose_cols else None

    if y_gluc:
        _dual_axis_chart(df, y_sleep, y_gluc, C["sleep"], C["glucose"], smooth, key="lf_sleep_dual")
    else:
        sm = df[y_sleep].rolling(smooth, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=sm, name=_label(y_sleep),
                                 line=dict(color=C["sleep"], width=2.5)))
        fig.update_layout(yaxis=dict(title=_label(y_sleep)), height=300)
        chart(fig, key="lf_sleep_single")

    # Sleep stage breakdown
    stage_cols = [c for c in ["prev_night_deep_sleep_min", "prev_night_rem_sleep_min"] if c in df.columns]
    if stage_cols:
        st.markdown("#### Sleep Stage Duration")
        colors = {"prev_night_deep_sleep_min": C["sleep"], "prev_night_rem_sleep_min": C["accent_soft"]}
        fig2 = go.Figure()
        for col in stage_cols:
            fig2.add_trace(go.Bar(
                x=df["date"], y=df[col],
                name=_label(col),
                marker_color=colors.get(col, C["chart1"]),
            ))
        fig2.update_layout(barmode="stack", yaxis=dict(title="Minutes"), height=260)
        chart(fig2, key="lf_sleep_stages")

    # Per-night glucose by sleep stage (Phase B integration)
    stage_cols = [
        "session_glucose_deep_mean",
        "session_glucose_light_mean",
        "session_glucose_rem_mean",
        "session_glucose_awake_mean",
    ]
    have = [c for c in stage_cols if c in df.columns]
    if have:
        st.subheader("Per-night glucose by sleep stage")
        sub = df[["date"] + have].dropna(how="all", subset=have)
        if not sub.empty:
            fig = go.Figure()
            for c in have:
                pretty = c.replace("session_glucose_", "").replace("_mean", "").title()
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub[c], mode="lines+markers", name=pretty,
                    connectgaps=False,
                ))
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Glucose (mg/dL)",
                height=320, margin=dict(l=10, r=10, t=10, b=10),
            )
            chart(fig)
        else:
            st.info("No nights with stage-tagged glucose yet.")


def _activity(df: pd.DataFrame) -> None:
    """Copied verbatim from app/main.py:_lifestyle_activity."""
    st.markdown("#### Activity & Readiness vs Glucose")
    smooth = st.session_state.get("smooth_window", 7)
    act_cols = _avail(df, "Activity")
    glucose_cols = _avail(df, "Glucose")
    if not act_cols:
        st.info("No activity data available.")
        return

    c1, c2 = st.columns(2)
    y_act = c1.selectbox("Activity metric", act_cols, format_func=_label, key="act_y1")
    y_gluc = c2.selectbox("Glucose metric", glucose_cols, format_func=_label, key="act_y2") if glucose_cols else None

    if y_gluc:
        _dual_axis_chart(df, y_act, y_gluc, C["activity"], C["glucose"], smooth, key="lf_act_dual")
    else:
        sm = df[y_act].rolling(smooth, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=sm, name=_label(y_act),
                                 line=dict(color=C["activity"], width=2.5)))
        fig.update_layout(yaxis=dict(title=_label(y_act)), height=300)
        chart(fig, key="lf_act_single")

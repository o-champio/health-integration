"""Cross-page Streamlit helpers shared by pages/*.py.

Page-specific helpers stay in the page module that owns them. Move a helper
here only when at least two pages use it.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Category → column groups ──────────────────────────────────────────────────

_CAT_COLS: dict[str, list[str]] = {
    "Glucose": [
        "glucose_mean", "glucose_tir", "glucose_tbr", "glucose_tar",
        "glucose_cv", "glucose_gmi", "glucose_std", "glucose_min", "glucose_max",
    ],
    "Sleep": [
        "prev_night_hrv", "prev_night_sleep_score", "prev_night_deep_sleep_min",
        "prev_night_rem_sleep_min", "prev_night_total_sleep_min",
        "prev_night_lowest_hr", "prev_night_efficiency", "prev_night_restless",
    ],
    "Activity": [
        "prev_day_activity_score", "prev_day_steps", "prev_day_active_calories",
        "prev_day_high_activity_min", "prev_day_readiness_score",
    ],
    "Stress": [
        "prev_day_stress_high", "prev_day_recovery_high", "prev_day_body_temp_dev",
    ],
    "Derived": [
        "sleep_activity_ratio", "hrv_hr_ratio", "glucose_mean_7d", "glucose_cv_7d",
    ],
}


def _avail(df: pd.DataFrame, cat: str, min_obs: int = 5) -> list[str]:
    return [c for c in _CAT_COLS.get(cat, []) if c in df.columns and df[c].notna().sum() > min_obs]


# ── Labels ────────────────────────────────────────────────────────────────────

_LABEL_FIXES = {
    "Tir": "TIR", "Tbr": "TBR", "Tar": "TAR",
    " Cv": " CV", "Gmi": "GMI", "Hrv": "HRV", " Hr": " HR",
}


def _label(col: str) -> str:
    s = (
        col
        .replace("glucose_", "Glucose ")
        .replace("prev_night_", "Sleep ")
        .replace("prev_day_", "Activity ")
        .replace("session_", "Session ")
        .replace("sleep_", "Sleep ")
        .replace("readiness_", "Readiness ")
        .replace("activity_", "Activity ")
        .replace("stress_", "Stress ")
        .replace("contributors.", "")
        .replace("_", " ")
        .title()
    )
    for wrong, right in _LABEL_FIXES.items():
        s = s.replace(wrong, right)
    return s


def _filter_raw(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Clip raw glucose readings to the date range of df."""
    if raw is None or raw.empty or df.empty:
        return raw
    lo = df["date"].min()
    hi = df["date"].max() + pd.Timedelta(days=1)
    return raw[(raw["timestamp"] >= lo) & (raw["timestamp"] < hi)].copy()


def _filter_events(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Clip events to the date range of df."""
    if events is None or events.empty or df.empty:
        return pd.DataFrame(columns=["timestamp", "event_type", "value"])
    lo = df["date"].min()
    hi = df["date"].max() + pd.Timedelta(days=1)
    return events[(events["timestamp"] >= lo) & (events["timestamp"] < hi)].copy()


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the date-range filter + settings. Returns the filtered dataframe.

    Navigation (page switcher) is handled by `st.navigation` in `main()`.
    """
    st.sidebar.markdown("### Health Dashboard")
    st.sidebar.markdown("---")

    dates = pd.to_datetime(df["date"])
    min_date = dates.min().date()
    max_date = dates.max().date()

    if "preset_start" not in st.session_state:
        st.session_state.preset_start = max(min_date, max_date - timedelta(days=90))

    st.sidebar.markdown("**Date range**")
    presets = [("1W", 7), ("2W", 14), ("MTD", 0), ("1M", 30), ("3M", 90), ("6M", 180), ("All", -1)]
    cols4 = st.sidebar.columns(4)
    for i, (lbl, days) in enumerate(presets):
        if cols4[i % 4].button(lbl, key=f"pr_{lbl}", use_container_width=True):
            if days == -1:
                st.session_state.preset_start = min_date
            elif days == 0:
                st.session_state.preset_start = max_date.replace(day=1)
            else:
                st.session_state.preset_start = max(min_date, max_date - timedelta(days=days))

    start = st.sidebar.date_input(
        "From", value=st.session_state.preset_start,
        min_value=min_date, max_value=max_date,
    )
    end = st.sidebar.date_input(
        "To", value=max_date,
        min_value=min_date, max_value=max_date,
    )
    st.session_state.preset_start = start

    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    filtered = df[mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Days in range", len(filtered))

    with st.sidebar.expander("⚙ Settings"):
        st.session_state["smooth_window"] = st.slider(
            "Smoothing window (days)", 1, 30,
            st.session_state.get("smooth_window", 7),
        )
        st.session_state["corr_method"] = st.selectbox(
            "Correlation method", ["spearman", "pearson"],
            index=0 if st.session_state.get("corr_method", "spearman") == "spearman" else 1,
        )

    return filtered


def _dual_axis_chart(
    df: pd.DataFrame,
    y1: str,
    y2: str,
    color1: str,
    color2: str,
    smooth: int,
    height: int = 320,
    key: str = "",
) -> None:
    if y1 not in df.columns or y2 not in df.columns:
        return
    s1 = df[y1].rolling(smooth, min_periods=1).mean()
    s2 = df[y2].rolling(smooth, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=s1, name=_label(y1), yaxis="y1",
        line=dict(color=color1, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=s2, name=_label(y2), yaxis="y2",
        line=dict(color=color2, width=2, dash="dot"),
    ))
    fig.update_layout(
        yaxis=dict(
            title=dict(text=_label(y1), font=dict(color=color1)),
            tickfont=dict(color=color1),
        ),
        yaxis2=dict(
            title=dict(text=_label(y2), font=dict(color=color2)),
            tickfont=dict(color=color2),
            overlaying="y", side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True, key=key or f"dual_{y1}_{y2}")

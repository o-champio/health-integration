"""Workout Analysis — profiles and glucose response curve.

Phase C drop list:
- Delta Analysis tab (overlaps with Response Curve)
- Nadir & Timing tab (overlaps with Response Curve)
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.processing.workout_glucose import (
    build_workout_glucose_df,
    glucose_response_curve,
    workout_summary_by_type,
)
from app._shared import _filter_raw, chart, tabs_or_select
from app._theme import C


def _profiles(wg: pd.DataFrame, summary: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workouts", len(wg))
    c2.metric("Activity Types", wg["activity"].nunique())
    c3.metric("Avg Duration", f"{wg['duration_min'].mean():.0f} min")
    avg_delta = wg["delta_during"].mean()
    c4.metric("Avg Glucose Change", f"{avg_delta:+.0f} mg/dL" if pd.notna(avg_delta) else "—")

    if not summary.empty:
        st.markdown("#### Summary by Activity Type")
        disp_summary = summary.rename(columns={
            "activity": "Activity", "workouts": "N", "avg_duration": "Avg Duration (min)",
            "avg_calories": "Avg Calories", "avg_pre": "Avg Pre",
            "avg_delta_during": "Avg Δ During", "avg_delta_post": "Avg Δ Post (1h)",
            "avg_nadir": "Avg Nadir", "avg_nadir_time": "Avg Nadir (min after)",
        })
        st.dataframe(disp_summary, use_container_width=True, hide_index=True)

    st.markdown("#### All Workouts")
    disp_wg = wg[[
        "activity", "day", "time_of_day", "duration_min", "calories",
        "pre_avg", "during_avg", "post_60_avg", "delta_during", "delta_post",
        "nadir_post_120", "nadir_time_min",
    ]].copy()
    disp_wg.columns = [
        "Activity", "Date", "Time of Day", "Duration (min)", "Calories",
        "Pre Avg", "During Avg", "Post 1h Avg", "Δ During", "Δ Post 1h",
        "Nadir (2h)", "Nadir (min after)",
    ]
    st.dataframe(disp_wg, use_container_width=True, hide_index=True)


def _response_curve(curve: pd.DataFrame) -> None:
    if curve.empty:
        st.info("Not enough glucose readings around workouts to plot a curve.")
    else:
        # Average curve per activity type
        avg_curve = (
            curve.groupby(["activity", "relative_min"])["glucose_delta"]
            .mean()
            .reset_index()
        )

        fig = go.Figure()
        colors = [C["chart1"], C["chart2"], C["chart3"], C["danger"], C["warning"]]
        for i, act in enumerate(avg_curve["activity"].unique()):
            act_data = avg_curve[avg_curve["activity"] == act]
            n_workouts = curve[curve["activity"] == act]["workout_idx"].nunique()
            fig.add_trace(go.Scatter(
                x=act_data["relative_min"],
                y=act_data["glucose_delta"],
                mode="lines+markers",
                name=f"{act} (n={n_workouts})",
                line=dict(color=colors[i % len(colors)], width=2.5),
                marker=dict(size=4),
            ))

        fig.add_vline(x=0, line_dash="dash", line_color=C["text_muted"], annotation_text="Start")
        fig.add_hline(y=0, line_dash="dot", line_color=C["border"])
        fig.update_layout(
            xaxis_title="Minutes relative to workout start",
            yaxis_title="Glucose change from baseline (mg/dL)",
            height=450,
            margin=dict(t=30, b=30),
        )
        chart(fig, key="workout_response_curve")

        # Individual traces (faded) per activity
        with st.expander("Individual workout traces"):
            fig2 = go.Figure()
            for act in curve["activity"].unique():
                act_data = curve[curve["activity"] == act]
                for widx in act_data["workout_idx"].unique():
                    trace = act_data[act_data["workout_idx"] == widx]
                    fig2.add_trace(go.Scatter(
                        x=trace["relative_min"],
                        y=trace["glucose_delta"],
                        mode="lines",
                        name=f"{act} #{widx}",
                        opacity=0.4,
                        showlegend=False,
                    ))
            fig2.add_vline(x=0, line_dash="dash", line_color=C["text_muted"])
            fig2.add_hline(y=0, line_dash="dot", line_color=C["border"])
            fig2.update_layout(
                xaxis_title="Minutes relative to workout start",
                yaxis_title="Glucose change (mg/dL)",
                height=400,
                margin=dict(t=20, b=20),
            )
            chart(fig2, key="workout_individual_traces")


def render(df: pd.DataFrame, raw_glucose: pd.DataFrame,
           workouts: pd.DataFrame) -> None:
    st.markdown("## Workout Analysis")
    st.caption("Glucose response before, during, and after exercise — by activity type.")
    st.caption(
        "💡 `unknown` = Apple Watch session pulled via Dexcom (≥25 min). "
        "No activity type, calories, or HR available for those rows."
    )

    if workouts.empty:
        st.warning("No workout data available. Ensure your Oura token is configured.")
        return

    raw = _filter_raw(raw_glucose, df)
    if raw.empty:
        st.warning("No glucose data in the selected date range.")
        return

    wg = build_workout_glucose_df(raw, workouts)
    if wg.empty:
        st.info("No workouts with overlapping glucose data found in the selected range.")
        return

    summary = workout_summary_by_type(wg)
    curve = glucose_response_curve(raw, workouts)

    labels = ["Workout Profiles", "Glucose Response Curve"]
    chosen = tabs_or_select(labels)
    if chosen:  # mobile
        if chosen == "Workout Profiles": _profiles(wg, summary)
        elif chosen == "Glucose Response Curve": _response_curve(curve)
    else:  # desktop
        tab1, tab2 = st.tabs(labels)
        with tab1: _profiles(wg, summary)
        with tab2: _response_curve(curve)

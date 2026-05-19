"""Glucose Deep Dive — trends, TIR breakdown, hourly heatmap, insulin/meal annotation."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app._shared import _filter_raw, _filter_events, chart
from app._theme import C


def render(df: pd.DataFrame, raw_glucose: pd.DataFrame,
           events: pd.DataFrame | None) -> None:
    raw = _filter_raw(raw_glucose, df)
    ev = _filter_events(events, df) if events is not None else pd.DataFrame()

    tab_trends, tab_tir, tab_hourly, tab_meals = st.tabs(
        ["Trends & GMI", "TIR Breakdown", "Hourly Patterns", "Insulin & Meals"]
    )
    with tab_trends:
        _trends(df, ev)
    with tab_tir:
        _tir_breakdown(df)
    with tab_hourly:
        _hourly(raw)
    with tab_meals:
        _insulin_meals(df, ev)


# ── Private helper ─────────────────────────────────────────────────────────────

def _dawn_rise(raw: pd.DataFrame) -> float | None:
    if raw is None or raw.empty:
        return None
    g = raw.copy()
    g["hour"] = g["timestamp"].dt.hour
    baseline = g[g["hour"].between(0, 2)]["glucose_mgdl"].mean()
    peak = g[g["hour"].between(5, 8)]["glucose_mgdl"].mean()
    if pd.isna(baseline) or pd.isna(peak):
        return None
    return float(peak - baseline)


# ── Tab renderers ──────────────────────────────────────────────────────────────

def _trends(df: pd.DataFrame, events: pd.DataFrame) -> None:
    """Copied verbatim from app/main.py:_glucose_trends."""
    st.markdown("#### Glucose Mean Trend")
    smooth = st.session_state.get("smooth_window", 7)
    if "glucose_mean" not in df.columns:
        st.info("No glucose data available.")
        return

    sm = df["glucose_mean"].rolling(smooth, min_periods=1).mean()
    fig = go.Figure()
    fig.add_hrect(
        y0=70, y1=180, fillcolor="rgba(34,197,94,0.06)",
        line=dict(width=0), annotation_text="70–180 mg/dL", annotation_position="right",
    )
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["glucose_mean"],
        name="Daily mean", mode="markers",
        marker=dict(color=C["glucose"], size=5, opacity=0.45),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=sm,
        name=f"{smooth}d avg", line=dict(color=C["glucose"], width=2.5),
    ))

    # Meal and insulin event markers (daily aggregates from events log)
    if events is not None and not events.empty:
        meal_days = events[events["event_type"] == "food"]["timestamp"].dt.normalize().value_counts()
        insulin_days = events[events["event_type"].isin(["insulin_rapid", "insulin_long"])]["timestamp"].dt.normalize().value_counts()
        if not meal_days.empty:
            meal_y = df.set_index("date")["glucose_mean"].reindex(meal_days.index).values
            fig.add_trace(go.Scatter(
                x=meal_days.index, y=meal_y,
                name="Meal logged", mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=C["warning"], opacity=0.8),
            ))
        if not insulin_days.empty:
            ins_y = df.set_index("date")["glucose_mean"].reindex(insulin_days.index).values
            fig.add_trace(go.Scatter(
                x=insulin_days.index, y=ins_y,
                name="Insulin logged", mode="markers",
                marker=dict(symbol="circle", size=9, color=C["danger"], opacity=0.75,
                            line=dict(width=1, color=C["text"])),
            ))

    fig.update_layout(yaxis=dict(title="mg/dL"), height=320)
    chart(fig, key="gl_trends")

    n_readings = df["glucose_readings"].sum() if "glucose_readings" in df.columns else None
    n_days = df["date"].nunique()
    if n_readings is not None and n_days > 0:
        avg_per_day = n_readings / n_days
        if avg_per_day < 10:
            st.caption(
                f"ℹ️ Low data density: avg {avg_per_day:.1f} readings/day. "
                "Hourly patterns and variability metrics improve with continuous CGM (Dexcom)."
            )

    if "glucose_gmi" not in df.columns:
        return
    st.markdown("#### Glucose Management Indicator (GMI)")
    st.caption("Estimates HbA1c from CGM data. Target for most T1D adults: < 7.0%.")
    gmi_sm = df["glucose_gmi"].rolling(smooth, min_periods=1).mean()
    c1, c2 = st.columns(2)
    latest_gmi = df["glucose_gmi"].dropna().iloc[-1] if df["glucose_gmi"].notna().any() else None
    avg_gmi = df["glucose_gmi"].mean()
    if latest_gmi is not None:
        c1.metric("Latest GMI", f"{latest_gmi:.2f}%",
                  delta=f"{latest_gmi - 7.0:+.2f}pp vs 7.0% target")
    if pd.notna(avg_gmi):
        c2.metric("Period Avg GMI", f"{avg_gmi:.2f}%")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["date"], y=gmi_sm, name="GMI",
        line=dict(color=C["warning"], width=2),
    ))
    fig2.add_hline(y=7.0, line_dash="dash", line_color=C["danger"],
                   annotation_text="7.0% target", annotation_font_color=C["danger"])
    fig2.add_hline(y=6.5, line_dash="dash", line_color=C["success"],
                   annotation_text="6.5% excellent", annotation_font_color=C["success"])
    fig2.update_layout(yaxis=dict(title="GMI (%)"), height=260)
    chart(fig2, key="gl_gmi")


def _tir_breakdown(df: pd.DataFrame) -> None:
    st.markdown("#### Average TIR Breakdown")
    needed = ["glucose_tir", "glucose_tar", "glucose_tbr"]
    if not all(c in df.columns for c in needed):
        st.info("TIR breakdown data not available.")
        return

    avg_tir = df["glucose_tir"].mean()
    avg_tar = df["glucose_tar"].mean()
    avg_tbr = df["glucose_tbr"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("In Range (70–180)", f"{avg_tir:.1%}",
              delta=f"{(avg_tir - 0.70)*100:+.1f}pp vs 70% target")
    c2.metric("Above Range (>180)", f"{avg_tar:.1%}")
    c3.metric("Below Range (<70)", f"{avg_tbr:.1%}")

    fig = go.Figure(go.Pie(
        labels=["In Range", "Above Range", "Below Range"],
        values=[avg_tir, avg_tar, avg_tbr],
        hole=0.62,
        marker_colors=[C["success"], C["danger"], C["warning"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
    ))
    fig.update_layout(height=280, showlegend=False)
    chart(fig, key="gl_tir_donut")

    st.markdown("#### Daily TIR Over Time")
    tir_data = df[["date"] + needed].dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=tir_data["date"], y=tir_data["glucose_tbr"],
                          name="Below", marker_color=C["warning"]))
    fig2.add_trace(go.Bar(x=tir_data["date"], y=tir_data["glucose_tir"],
                          name="In Range", marker_color=C["success"]))
    fig2.add_trace(go.Bar(x=tir_data["date"], y=tir_data["glucose_tar"],
                          name="Above", marker_color=C["danger"]))
    fig2.update_layout(
        barmode="stack",
        yaxis=dict(title="Fraction of day", tickformat=".0%"),
        height=300,
    )
    chart(fig2, key="gl_tir_daily")


def _hourly(raw: pd.DataFrame) -> None:
    """Copied verbatim from app/main.py:_glucose_hourly."""
    st.markdown("#### Average Glucose by Hour of Day")
    st.caption("Mean ± 1 SD across all CGM readings in the selected date range.")
    if raw is None or raw.empty:
        st.info("No raw glucose readings available.")
        return

    total_readings = len(raw)
    n_days = raw["timestamp"].dt.normalize().nunique()
    avg_per_day = total_readings / n_days if n_days > 0 else 0
    if avg_per_day < 4:
        st.warning(
            f"Only {avg_per_day:.1f} readings/day on average — not enough for meaningful hourly patterns. "
            "This chart will become useful once continuous CGM (Dexcom) is integrated."
        )
        return
    if avg_per_day < 15:
        st.caption(
            f"⚠️ Sparse data ({avg_per_day:.1f} readings/day). "
            "Patterns shown but reliability improves with continuous CGM."
        )

    g = raw.copy()
    g["hour"] = g["timestamp"].dt.hour
    h = (
        g.groupby("hour")["glucose_mgdl"]
        .agg(mean="mean", std="std")
        .reset_index()
        .fillna({"std": 0})
    )
    h["upper"] = h["mean"] + h["std"]
    h["lower"] = (h["mean"] - h["std"]).clip(lower=0)

    fig = go.Figure()
    # ±1 SD band
    fig.add_trace(go.Scatter(
        x=list(h["hour"]) + list(h["hour"])[::-1],
        y=list(h["upper"]) + list(h["lower"])[::-1],
        fill="toself",
        fillcolor="rgba(52,211,153,0.12)",
        line=dict(width=0),
        name="±1 SD",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=h["hour"], y=h["mean"],
        name="Mean glucose",
        line=dict(color=C["glucose"], width=2.5),
    ))
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(34,197,94,0.06)",
        line=dict(width=0),
    )
    fig.add_hline(y=70, line_dash="dash", line_color=C["warning"], line_width=1)
    fig.add_hline(y=180, line_dash="dash", line_color=C["danger"], line_width=1)
    fig.update_layout(
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Glucose (mg/dL)"),
        height=320,
    )
    chart(fig, key="gl_hourly")

    dawn = _dawn_rise(raw)
    if dawn is not None:
        if dawn > 15:
            st.markdown(
                f'<div class="alert-dawn">🌅 <b>Dawn phenomenon detected:</b> avg +{dawn:.0f} mg/dL rise from 0–2 AM baseline to 5–8 AM window.</div>',
                unsafe_allow_html=True,
            )
        elif dawn > 5:
            st.markdown(
                f'<div class="insight-card">🌅 Mild dawn rise: +{dawn:.0f} mg/dL avg from overnight to early morning.</div>',
                unsafe_allow_html=True,
            )


def _insulin_meals(df: pd.DataFrame, events: pd.DataFrame) -> None:
    """Copied verbatim from app/main.py:_glucose_insulin_meals.

    Insulin & meal logging tab — shows available data, placeholders for future.
    """
    st.markdown("#### Insulin & Meal Events")

    has_rapid = not events.empty and (events["event_type"] == "insulin_rapid").any()
    has_long = not events.empty and (events["event_type"] == "insulin_long").any()
    has_food = not events.empty and (events["event_type"] == "food").any()
    any_events = has_rapid or has_long or has_food

    if not any_events:
        st.info(
            "No insulin or meal events logged in the selected range. "
            "Events come from LibreLink CSV exports (pre-cutover) and the Dexcom "
            "`/events` endpoint (post-cutover)."
        )
        return

    # Events are present — show what we have
    c1, c2, c3 = st.columns(3)
    if has_rapid:
        n = (events["event_type"] == "insulin_rapid").sum()
        c1.metric("Rapid Insulin Events", n)
    if has_long:
        n = (events["event_type"] == "insulin_long").sum()
        c2.metric("Long Insulin Events", n)
    if has_food:
        n = (events["event_type"] == "food").sum()
        c3.metric("Meal Events", n)

    # Timeline chart
    fig = go.Figure()
    colors = {"insulin_rapid": C["danger"], "insulin_long": C["warning"], "food": C["activity"]}
    symbols = {"insulin_rapid": "circle", "insulin_long": "diamond", "food": "triangle-up"}
    labels = {"insulin_rapid": "Rapid insulin", "insulin_long": "Long insulin", "food": "Meal"}

    for etype in ["insulin_rapid", "insulin_long", "food"]:
        sub = events[events["event_type"] == etype]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=sub["value"],
            name=labels[etype],
            mode="markers",
            marker=dict(
                symbol=symbols[etype],
                size=9,
                color=colors[etype],
                opacity=0.8,
            ),
            hovertemplate=f"{labels[etype]}: %{{y}} | %{{x}}<extra></extra>",
        ))

    ylab = "Units / Grams"
    fig.update_layout(
        yaxis=dict(title=ylab),
        xaxis=dict(title="Date"),
        height=320,
        legend=dict(orientation="h", y=1.08),
    )
    chart(fig, key="gl_events")

    # Daily insulin totals if values present
    if has_rapid:
        rapid = events[events["event_type"] == "insulin_rapid"].copy()
        rapid["date"] = rapid["timestamp"].dt.normalize()
        daily_rapid = rapid.groupby("date")["value"].sum().reset_index()
        daily_rapid.columns = ["date", "rapid_units"]
        if daily_rapid["rapid_units"].notna().any():
            st.markdown("#### Daily Rapid Insulin Total")
            fig2 = go.Figure(go.Bar(
                x=daily_rapid["date"], y=daily_rapid["rapid_units"],
                marker_color=C["danger"], name="Rapid insulin (units)",
            ))
            fig2.update_layout(yaxis=dict(title="Units"), height=240)
            chart(fig2, key="gl_rapid_daily")

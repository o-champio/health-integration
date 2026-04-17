"""Health Dashboard -- T1D-focused Streamlit application.

Launch:
    streamlit run app/main.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.analysis import correlation_matrix, dual_correlation, run_multi_target_regression
from src.processing.features import build_analysis_df, get_feature_columns
from src.processing.pipeline import sync_all
from src.processing.workout_glucose import (
    build_workout_glucose_df,
    glucose_response_curve,
    workout_summary_by_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Palette ───────────────────────────────────────────────────────────────────

C: dict[str, str] = {
    "bg": "#0F172A",
    "card": "#111827",
    "surface": "#1E293B",
    "border": "#1F2937",
    "text": "#E5E7EB",
    "text_sec": "#9CA3AF",
    "text_muted": "#64748B",
    "primary": "#6C63FF",
    "accent": "#818CF8",
    "accent_soft": "#A5B4FC",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "sleep": "#818CF8",
    "activity": "#22D3EE",
    "glucose": "#34D399",
    "chart1": "#6C63FF",
    "chart2": "#22D3EE",
    "chart3": "#34D399",
    "pos": "#22C55E",
    "neg": "#EF4444",
    "neutral": "#64748B",
}

_AXIS_STYLE = dict(
    gridcolor=C["border"],
    linecolor=C["border"],
    zerolinecolor=C["border"],
    tickfont=dict(color=C["text_sec"], size=12),
    title_font=dict(color=C["text_sec"], size=13),
    title_standoff=10,
)
pio.templates["health"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["text"], family="Inter, system-ui, sans-serif", size=13),
        title_font=dict(color=C["text"], size=15, family="Inter, system-ui, sans-serif"),
        xaxis=_AXIS_STYLE,
        yaxis=_AXIS_STYLE,
        legend=dict(
            bgcolor=C["card"],
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text_sec"], size=12),
        ),
        colorway=[C["chart1"], C["chart2"], C["chart3"], C["warning"], C["danger"], C["accent"]],
        margin=dict(t=44, b=44, l=52, r=44),
        hoverlabel=dict(
            bgcolor=C["surface"],
            bordercolor=C["border"],
            font=dict(color=C["text"], size=13),
        ),
    )
)
pio.templates.default = "health"

_CSS = f"""
<style>
  /* ── Base ──────────────────────────────────────────────── */
  .stApp {{ background-color: {C['bg']}; color: {C['text']}; }}
  .main .block-container {{ padding-top: 1.5rem; }}

  /* ── Sidebar ───────────────────────────────────────────── */
  section[data-testid="stSidebar"] {{
    background-color: {C['card']};
    border-right: 1px solid {C['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

  /* ── General text ──────────────────────────────────────── */
  p, span, div, li {{ color: {C['text']}; }}
  h1, h2, h3, h4, h5, h6 {{ color: {C['text']} !important; font-weight: 700; }}
  small, .stCaption, .stCaption p, [data-testid="stCaptionContainer"] p {{
    color: {C['text_muted']} !important;
    font-size: 0.82rem;
  }}

  /* ── Form labels & inputs ──────────────────────────────── */
  label, .stSelectbox label, .stMultiSelect label,
  .stSlider label, .stDateInput label, .stTextInput label,
  .stRadio label, .stCheckbox label {{
    color: {C['text_sec']} !important;
    font-size: 0.82rem;
    font-weight: 500;
  }}
  /* selectbox / multiselect pill area */
  [data-baseweb="select"] [data-baseweb="tag"] {{ background: {C['surface']} !important; }}
  [data-baseweb="select"] span {{ color: {C['text']} !important; }}
  /* dropdown options list */
  [data-baseweb="menu"] li {{ color: {C['text']} !important; background: {C['surface']} !important; }}
  [data-baseweb="menu"] li:hover {{ background: {C['border']} !important; }}
  /* date input text */
  [data-testid="stDateInput"] input {{ color: {C['text']} !important; background: {C['surface']} !important; }}

  /* ── Metrics ───────────────────────────────────────────── */
  [data-testid="stMetric"] {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"] p {{
    color: {C['text_sec']} !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  [data-testid="stMetricValue"] {{
    color: {C['text']} !important;
    font-size: 1.6rem;
    font-weight: 700;
  }}
  [data-testid="stMetricDelta"] svg {{ vertical-align: middle; }}
  [data-testid="stMetricDelta"] > div {{ color: {C['text_sec']} !important; font-size: 0.82rem; }}

  /* ── Tabs ──────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {C['card']};
    border-bottom: 1px solid {C['border']};
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {C['text_sec']} !important;
    background: transparent !important;
    border-bottom: 2px solid transparent;
    padding: 8px 18px;
    font-size: 0.88rem;
    font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C['primary']} !important;
    border-bottom-color: {C['primary']} !important;
  }}

  /* ── Buttons ───────────────────────────────────────────── */
  .stButton > button {{
    background: {C['card']};
    color: {C['text']} !important;
    border: 1px solid {C['border']};
    border-radius: 8px;
    font-size: 0.8rem;
    padding: 4px 8px;
    transition: all 0.15s;
  }}
  .stButton > button:hover {{
    background: {C['primary']};
    border-color: {C['primary']};
    color: #fff !important;
  }}

  /* ── Expanders ─────────────────────────────────────────── */
  [data-testid="stExpander"] summary {{
    background: {C['surface']} !important;
    border: 1px solid {C['border']};
    border-radius: 8px;
    color: {C['text']} !important;
  }}
  [data-testid="stExpander"] summary:hover {{ background: {C['border']} !important; }}
  [data-testid="stExpander"] summary span, [data-testid="stExpander"] summary p {{
    color: {C['text']} !important;
  }}
  [data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-top: none;
    border-radius: 0 0 8px 8px;
  }}

  /* ── Alerts / info boxes ───────────────────────────────── */
  [data-testid="stAlert"] {{
    background: {C['surface']} !important;
    border-radius: 8px;
  }}
  [data-testid="stAlert"] p {{ color: {C['text']} !important; }}

  /* ── Dataframes ────────────────────────────────────────── */
  [data-testid="stDataFrame"] {{ border: 1px solid {C['border']}; border-radius: 8px; }}
  .stDataFrame thead tr th {{
    background: {C['surface']} !important;
    color: {C['text_sec']} !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .stDataFrame tbody tr td {{ color: {C['text']} !important; background: {C['card']} !important; }}
  .stDataFrame tbody tr:hover td {{ background: {C['surface']} !important; }}

  /* ── Plotly chart container ────────────────────────────── */
  .js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
  .js-plotly-plot .plotly .modebar-btn path {{ fill: {C['text_muted']} !important; }}

  /* ── Slider ────────────────────────────────────────────── */
  [data-testid="stSlider"] [data-testid="stMarkdown"] p {{ color: {C['text_sec']} !important; }}

  /* ── Divider ───────────────────────────────────────────── */
  hr {{ border-color: {C['border']}; opacity: 0.4; }}

  /* ── Custom alert cards ────────────────────────────────── */
  .alert-hypo {{
    background: rgba(239,68,68,0.12);
    border: 1px solid {C['danger']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
  .alert-dawn {{
    background: rgba(245,158,11,0.12);
    border: 1px solid {C['warning']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
  .insight-card {{
    background: rgba(108,99,255,0.12);
    border: 1px solid {C['primary']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
  .success-card {{
    background: rgba(34,197,94,0.12);
    border: 1px solid {C['success']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
</style>
"""


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


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Syncing all data…")
def _sync_all() -> dict[str, pd.DataFrame]:
    """Single incremental sync — glucose, daily, workouts, high-freq.

    TTL=1h since incremental sync is fast (~4s when cached).
    Use the 'Sync now' sidebar button to force a refresh.
    """
    return sync_all()


def _load_analysis(synced: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_analysis_df(synced["daily"])


def _load_raw_glucose(synced: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return synced["glucose"]


def _load_workouts_from_sync(synced: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return synced["workouts"]


@st.cache_data(ttl=43200)
def _load_events() -> pd.DataFrame:
    """Load insulin and food events from LibreLink CSVs.

    Returns DataFrame with columns: timestamp, event_type, value
      event_type: 'insulin_rapid' | 'insulin_long' | 'food'
      value: units (insulin) or grams (food), NaN when not logged

    When Dexcom API or dedicated meal-logging is integrated, replace this
    loader while keeping the same output schema so downstream charts don't break.
    """
    try:
        from src.api.libre_client import load_all
        raw = load_all()
        frames = []
        type_map = {3: "insulin_rapid", 4: "food", 5: "insulin_long"}
        value_cols = {
            3: ["Rapid-Acting Insulin (units)", "Rapid Acting Insulin (units)"],
            4: ["Carbohydrates (grams)", "Carbs (grams)"],
            5: ["Long-Acting Insulin Value (units)", "Long Acting Insulin (units)"],
        }
        for rtype, etype in type_map.items():
            sub = raw[raw["Record Type"] == rtype].copy()
            if sub.empty:
                continue
            sub = sub.rename(columns={"Device Timestamp": "timestamp"})[["timestamp"]].copy()
            sub["event_type"] = etype
            # Try known column names for the value; fall back to NaN
            val = np.nan
            for cname in value_cols.get(rtype, []):
                if cname in raw.columns:
                    val = pd.to_numeric(raw.loc[sub.index, cname], errors="coerce")
                    break
            sub["value"] = val
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "event_type", "value"])
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        log.warning("Could not load events: %s", exc)
        return pd.DataFrame(columns=["timestamp", "event_type", "value"])


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sync all pipelines once and return (analysis_df, raw_glucose, workouts)."""
    try:
        synced = _sync_all()
        return (
            _load_analysis(synced),
            _load_raw_glucose(synced),
            _load_workouts_from_sync(synced),
        )
    except FileNotFoundError as exc:
        st.error(f"Data not found: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        log.exception("Pipeline failed")
        st.stop()


def _filter_raw(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Clip raw glucose readings to the date range of df."""
    if raw is None or raw.empty or df.empty:
        return raw
    lo = df["date"].min()
    hi = df["date"].max() + pd.Timedelta(days=1)
    return raw[(raw["timestamp"] >= lo) & (raw["timestamp"] < hi)].copy()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    st.sidebar.markdown("### Health Dashboard")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Glucose Deep Dive", "Lifestyle Factors", "Workout Analysis", "Correlation Explorer", "Regression & Insights"],
        label_visibility="collapsed",
    )
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
    if "glucose_tir" in filtered.columns:
        avg_tir = filtered["glucose_tir"].mean()
        if pd.notna(avg_tir):
            st.sidebar.metric("Avg TIR", f"{avg_tir:.1%}")
    if "glucose_cv" in filtered.columns:
        avg_cv = filtered["glucose_cv"].mean()
        if pd.notna(avg_cv):
            st.sidebar.metric("Glucose CV", f"{avg_cv:.2f}")
    if "prev_night_hrv" in filtered.columns:
        avg_hrv = filtered["prev_night_hrv"].mean()
        if pd.notna(avg_hrv):
            st.sidebar.metric("Avg HRV", f"{avg_hrv:.0f} ms")

    with st.sidebar.expander("⚙ Settings"):
        st.session_state["smooth_window"] = st.slider(
            "Smoothing window (days)", 1, 30,
            st.session_state.get("smooth_window", 7),
        )
        st.session_state["corr_method"] = st.selectbox(
            "Correlation method", ["spearman", "pearson"],
            index=0 if st.session_state.get("corr_method", "spearman") == "spearman" else 1,
        )

    return filtered, page


# ── Shared chart helper ───────────────────────────────────────────────────────

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


# ── Overview ──────────────────────────────────────────────────────────────────

def _page_overview(df: pd.DataFrame, raw_glucose: pd.DataFrame) -> None:
    st.markdown("## Today's Snapshot")
    raw = _filter_raw(raw_glucose, df)

    # Latest values
    latest_g = df.dropna(subset=["glucose_tir"]).iloc[-1] if "glucose_tir" in df.columns and df["glucose_tir"].notna().any() else None
    latest_s = df.dropna(subset=["prev_night_sleep_score"]).iloc[-1] if "prev_night_sleep_score" in df.columns and df["prev_night_sleep_score"].notna().any() else None
    latest_r = df.dropna(subset=["prev_day_readiness_score"]).iloc[-1] if "prev_day_readiness_score" in df.columns and df["prev_day_readiness_score"].notna().any() else None
    latest_a = df.dropna(subset=["prev_day_activity_score"]).iloc[-1] if "prev_day_activity_score" in df.columns and df["prev_day_activity_score"].notna().any() else None

    def _v(row, col, fmt="{:.0f}"):
        if row is None or col not in row.index:
            return "—"
        val = row[col]
        return fmt.format(val) if pd.notna(val) else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    tir_val = latest_g["glucose_tir"] if latest_g is not None and pd.notna(latest_g.get("glucose_tir")) else None
    c1.metric("Time in Range", f"{tir_val:.1%}" if tir_val is not None else "—")
    c2.metric("Mean Glucose", f"{_v(latest_g, 'glucose_mean')} mg/dL")
    c3.metric("Sleep Score", _v(latest_s, "prev_night_sleep_score"))
    c4.metric("Readiness", _v(latest_r, "prev_day_readiness_score"))
    c5.metric("HRV", f"{_v(latest_s, 'prev_night_hrv')} ms")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### Last 7 Days — Time in Range")
        last7 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=6)].copy()
        if "glucose_tir" in last7.columns and last7["glucose_tir"].notna().sum() > 0:
            colors = [
                C["success"] if v >= 0.70 else (C["warning"] if v >= 0.50 else C["danger"])
                for v in last7["glucose_tir"].fillna(0)
            ]
            fig = go.Figure(go.Bar(
                x=last7["date"],
                y=last7["glucose_tir"],
                marker_color=colors,
                text=[f"{v:.0%}" for v in last7["glucose_tir"].fillna(0)],
                textposition="outside",
                textfont=dict(color=C["text_sec"]),
            ))
            fig.add_hline(
                y=0.70, line_dash="dash", line_color=C["success"],
                annotation_text="70% target", annotation_font_color=C["success"],
            )
            fig.update_layout(
                yaxis=dict(title="TIR", tickformat=".0%", range=[0, 1.15]),
                height=250, showlegend=False, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, key="ov_tir_bar")
        else:
            st.info("No TIR data in the last 7 days.")

    with col_r:
        st.markdown("#### Alerts")
        _overview_alerts(df, raw)

    st.markdown("---")
    st.markdown("#### Correlation Highlights")
    _overview_insights(df)


def _overview_alerts(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    if "glucose_tbr" in df.columns:
        total = df["glucose_tbr"].notna().sum()
        hypo_days = (df["glucose_tbr"] > 0.01).sum()
        if hypo_days > 0 and total > 0:
            pct = hypo_days / total * 100
            st.markdown(
                f'<div class="alert-hypo">⚠️ <b>Hypo days:</b> {hypo_days}/{total} ({pct:.0f}%) had TBR &gt; 1%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="success-card">✓ <b>No hypoglycemia</b> detected in this period</div>',
                unsafe_allow_html=True,
            )

    dawn = _dawn_rise(raw)
    if dawn is not None:
        if dawn > 15:
            st.markdown(
                f'<div class="alert-dawn">🌅 <b>Dawn phenomenon:</b> avg +{dawn:.0f} mg/dL rise (0–2 AM → 5–8 AM)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="success-card">✓ <b>Dawn rise</b> mild (+{dawn:.0f} mg/dL)</div>',
                unsafe_allow_html=True,
            )

    if "glucose_cv" in df.columns:
        high_var = (df["glucose_cv"] > 0.36).sum()
        if high_var > 0:
            st.markdown(
                f'<div class="alert-hypo">📊 <b>High variability:</b> {high_var} days with CV &gt; 0.36</div>',
                unsafe_allow_html=True,
            )


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


def _overview_insights(df: pd.DataFrame) -> None:
    insights: list[tuple[str, str]] = []

    if "prev_night_total_sleep_min" in df.columns and "glucose_tir" in df.columns:
        sub = df[["prev_night_total_sleep_min", "glucose_tir"]].dropna()
        if len(sub) >= 10:
            good = sub[sub["prev_night_total_sleep_min"] >= 420]["glucose_tir"].mean()
            poor = sub[sub["prev_night_total_sleep_min"] < 420]["glucose_tir"].mean()
            if pd.notna(good) and pd.notna(poor) and abs(good - poor) > 0.03:
                pp = (good - poor) * 100
                card = "success-card" if pp > 0 else "insight-card"
                insights.append((card, f"🛌 TIR is <b>{pp:+.0f}pp</b> on 7+ hour sleep nights vs shorter nights"))

    if "prev_day_activity_score" in df.columns and "glucose_tir" in df.columns:
        sub = df[["prev_day_activity_score", "glucose_tir"]].dropna()
        if len(sub) >= 10:
            active = sub[sub["prev_day_activity_score"] >= 70]["glucose_tir"].mean()
            inactive = sub[sub["prev_day_activity_score"] < 70]["glucose_tir"].mean()
            if pd.notna(active) and pd.notna(inactive) and abs(active - inactive) > 0.03:
                pp = (active - inactive) * 100
                card = "success-card" if pp > 0 else "insight-card"
                insights.append((card, f"🏃 Active days (score ≥ 70) show <b>{pp:+.0f}pp</b> TIR vs less active days"))

    if "prev_night_hrv" in df.columns and "glucose_cv" in df.columns:
        sub = df[["prev_night_hrv", "glucose_cv"]].dropna()
        if len(sub) >= 10:
            r = sub["prev_night_hrv"].corr(sub["glucose_cv"])
            if pd.notna(r) and abs(r) > 0.2:
                direction = "lower" if r < 0 else "higher"
                card = "success-card" if r < 0 else "insight-card"
                insights.append((card, f"💓 Higher HRV correlates with <b>{direction}</b> glucose variability (r={r:.2f})"))

    if not insights:
        st.info("Not enough data for highlights yet.")
    else:
        for card_class, text in insights:
            st.markdown(f'<div class="{card_class}">{text}</div>', unsafe_allow_html=True)


# ── Glucose Deep Dive ─────────────────────────────────────────────────────────

def _page_glucose(
    df: pd.DataFrame,
    raw_glucose: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> None:
    raw = _filter_raw(raw_glucose, df)
    ev = _filter_events(events, df) if events is not None else pd.DataFrame()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Trends & GMI", "TIR Breakdown", "Hourly Patterns", "Variability", "Insulin & Meals"]
    )
    with tab1:
        _glucose_trends(df, ev)
    with tab2:
        _glucose_tir_breakdown(df)
    with tab3:
        _glucose_hourly(raw)
    with tab4:
        _glucose_variability(df)
    with tab5:
        _glucose_insulin_meals(df, ev)


def _filter_events(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Clip events to the date range of df."""
    if events is None or events.empty or df.empty:
        return pd.DataFrame(columns=["timestamp", "event_type", "value"])
    lo = df["date"].min()
    hi = df["date"].max() + pd.Timedelta(days=1)
    return events[(events["timestamp"] >= lo) & (events["timestamp"] < hi)].copy()


def _glucose_trends(df: pd.DataFrame, events: pd.DataFrame | None = None) -> None:
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
    st.plotly_chart(fig, use_container_width=True, key="gl_trends")

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
    st.plotly_chart(fig2, use_container_width=True, key="gl_gmi")


def _glucose_tir_breakdown(df: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True, key="gl_tir_donut")

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
    st.plotly_chart(fig2, use_container_width=True, key="gl_tir_daily")


def _glucose_hourly(raw: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True, key="gl_hourly")

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


def _glucose_variability(df: pd.DataFrame) -> None:
    st.markdown("#### Glucose Variability (CV)")
    st.caption("CV < 0.36 is the clinical target for glycemic stability.")
    if "glucose_cv" not in df.columns:
        st.info("No CV data available.")
        return

    smooth = st.session_state.get("smooth_window", 7)
    sm = df["glucose_cv"].rolling(smooth, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["glucose_cv"],
        name="Daily CV", mode="markers",
        marker=dict(color=C["chart2"], size=4, opacity=0.4),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=sm,
        name=f"{smooth}d rolling", line=dict(color=C["chart2"], width=2),
    ))
    fig.add_hline(y=0.36, line_dash="dash", line_color=C["warning"],
                  annotation_text="0.36 target")
    fig.update_layout(yaxis=dict(title="Coefficient of Variation"), height=300)
    st.plotly_chart(fig, use_container_width=True, key="gl_cv")


def _glucose_insulin_meals(df: pd.DataFrame, events: pd.DataFrame) -> None:
    """Insulin & meal logging tab — shows available data, placeholders for future."""
    st.markdown("#### Insulin & Meal Events")

    has_rapid = not events.empty and (events["event_type"] == "insulin_rapid").any()
    has_long = not events.empty and (events["event_type"] == "insulin_long").any()
    has_food = not events.empty and (events["event_type"] == "food").any()
    any_events = has_rapid or has_long or has_food

    if not any_events:
        st.info(
            "No insulin or meal events logged yet. "
            "Start logging in FreeStyle LibreLink, or connect a dedicated meal-logging app. "
            "Once available, this section will show:\n"
            "- Rapid & long-acting insulin doses over time\n"
            "- Meal timing and carb content\n"
            "- Post-meal glucose excursions\n"
            "- Insulin-on-board estimates\n"
            "- Carbohydrate-to-insulin ratio analysis"
        )
        st.markdown("---")
        st.markdown("##### Coming when Dexcom API is integrated")
        st.markdown(
            "- 5-minute glucose overlaid with insulin doses\n"
            "- Post-meal glucose peak and time-to-peak\n"
            "- Insulin sensitivity by time of day\n"
            "- Overnight basal rate assessment"
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
    st.plotly_chart(fig, use_container_width=True, key="gl_events")

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
            st.plotly_chart(fig2, use_container_width=True, key="gl_rapid_daily")

    st.caption(
        "💡 Future: once Dexcom 5-min data is available, this page will show "
        "post-meal excursions, insulin-on-board curves, and carb ratio analysis."
    )


# ── Lifestyle Factors ─────────────────────────────────────────────────────────

def _page_lifestyle(df: pd.DataFrame) -> None:
    tab1, tab2, tab3 = st.tabs(["Sleep", "Activity & Readiness", "HRV & Stress"])
    with tab1:
        _lifestyle_sleep(df)
    with tab2:
        _lifestyle_activity(df)
    with tab3:
        _lifestyle_hrv_stress(df)


def _lifestyle_sleep(df: pd.DataFrame) -> None:
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
        st.plotly_chart(fig, use_container_width=True, key="lf_sleep_single")

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
        st.plotly_chart(fig2, use_container_width=True, key="lf_sleep_stages")


def _lifestyle_activity(df: pd.DataFrame) -> None:
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
        st.plotly_chart(fig, use_container_width=True, key="lf_act_single")


def _lifestyle_hrv_stress(df: pd.DataFrame) -> None:
    st.markdown("#### HRV Over Time")
    smooth = st.session_state.get("smooth_window", 7)
    if "prev_night_hrv" not in df.columns:
        st.info("No HRV data available.")
        return

    sm = df["prev_night_hrv"].rolling(smooth, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["prev_night_hrv"],
        name="HRV", mode="markers",
        marker=dict(color=C["sleep"], size=4, opacity=0.35),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=sm,
        name=f"{smooth}d avg", line=dict(color=C["sleep"], width=2.5),
    ))
    fig.update_layout(yaxis=dict(title="HRV (ms)"), height=300)
    st.plotly_chart(fig, use_container_width=True, key="lf_hrv")

    stress_cols = _avail(df, "Stress")
    glucose_cols = _avail(df, "Glucose")
    if stress_cols:
        st.markdown("#### HRV vs Stress & Glucose")
        c1, c2 = st.columns(2)
        y_stress = c1.selectbox("Stress metric", stress_cols, format_func=_label, key="hrv_stress")
        _dual_axis_chart(df, "prev_night_hrv", y_stress, C["sleep"], C["warning"], smooth, key="lf_hrv_stress_dual")

    if glucose_cols:
        st.markdown("#### HRV vs Glucose")
        y_gluc = st.selectbox("Glucose metric", glucose_cols, format_func=_label, key="hrv_gluc")
        _dual_axis_chart(df, "prev_night_hrv", y_gluc, C["sleep"], C["glucose"], smooth, key="lf_hrv_gluc_dual")


# ── Correlation Explorer ──────────────────────────────────────────────────────

def _page_correlations(df: pd.DataFrame) -> None:
    tab1, tab2 = st.tabs(["Scatter Plot", "Correlation Heatmap"])
    with tab1:
        _corr_scatter(df)
    with tab2:
        _corr_heatmap(df)


def _corr_scatter(df: pd.DataFrame) -> None:
    st.markdown("#### Variable Selection")
    cats = list(_CAT_COLS.keys())

    c1, c2, c3, c4 = st.columns([2, 3, 2, 3])
    x_cat = c1.selectbox("X Category", cats, index=1, key="xc")
    x_cols = _avail(df, x_cat)
    if not x_cols:
        st.warning(f"No {x_cat} data in range.")
        return
    x_col = c2.selectbox("X Metric", x_cols, format_func=_label, key="xcol")

    y_cat = c3.selectbox("Y Category", cats, index=0, key="yc")
    y_cols = _avail(df, y_cat)
    if not y_cols:
        st.warning(f"No {y_cat} data in range.")
        return
    y_col = c4.selectbox("Y Metric", y_cols, format_func=_label, key="ycol")

    lag = st.slider("Lag X by N days (X leads Y)", 0, 3, 0, key="scatter_lag")
    method = st.session_state.get("corr_method", "spearman")

    x_series = df[x_col].shift(lag)
    scatter_df = pd.DataFrame({"x": x_series, "y": df[y_col]}).dropna()

    if len(scatter_df) < 5:
        st.warning("Not enough overlapping data for the selected pair.")
        return

    r_pearson = scatter_df["x"].corr(scatter_df["y"], method="pearson")
    r_spearman = scatter_df["x"].corr(scatter_df["y"], method="spearman")
    r_display = r_spearman if method == "spearman" else r_pearson
    trend_color = C["pos"] if r_display > 0.1 else (C["neg"] if r_display < -0.1 else C["neutral"])

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Pearson r", f"{r_pearson:.3f}")
    mc2.metric("Spearman ρ", f"{r_spearman:.3f}")
    mc3.metric("Observations", len(scatter_df))

    x_label = _label(x_col) + (f" (lag {lag}d)" if lag else "")
    fig = px.scatter(
        scatter_df, x="x", y="y",
        trendline="ols",
        trendline_color_override=trend_color,
        labels={"x": x_label, "y": _label(y_col)},
        opacity=0.55,
        color_discrete_sequence=[C["chart1"]],
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True, key="cx_scatter")


def _corr_heatmap(df: pd.DataFrame) -> None:
    st.markdown("#### Grouped Correlation Heatmap")
    method = st.session_state.get("corr_method", "spearman")
    cats = list(_CAT_COLS.keys())

    c1, c2 = st.columns(2)
    row_cats = c1.multiselect("Row categories", cats, default=["Glucose"], key="hm_rows")
    col_cats = c2.multiselect("Column categories", cats, default=["Sleep", "Activity", "Stress"], key="hm_cols")

    row_cols = [c for cat in row_cats for c in _avail(df, cat)]
    col_cols = [c for cat in col_cats for c in _avail(df, cat)]

    if not row_cols or not col_cols:
        st.info("Select at least one category for each axis.")
        return

    corr = correlation_matrix(df, row_cols, col_cols, method=method)
    if corr.empty:
        st.info("Not enough data to compute correlations.")
        return

    corr.index = [_label(c) for c in corr.index]
    corr.columns = [_label(c) for c in corr.columns]

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, C["neg"]], [0.5, C["neutral"]], [1, C["pos"]]],
        zmin=-1, zmax=1,
        aspect="auto",
        labels=dict(color="r"),
    )
    fig.update_layout(
        height=max(280, 52 * len(corr)),
        coloraxis_colorbar=dict(tickvals=[-1, -0.5, 0, 0.5, 1]),
    )
    st.plotly_chart(fig, use_container_width=True, key="cx_heatmap")

    # Top pairs table
    flat = corr.stack().reset_index()
    flat.columns = ["Y Metric", "X Metric", "r"]
    flat = flat[flat["Y Metric"] != flat["X Metric"]].sort_values("r", key=abs, ascending=False)
    st.markdown("**Strongest correlations**")
    st.dataframe(flat.head(10).reset_index(drop=True), use_container_width=True, hide_index=True)


# ── Regression & Insights ─────────────────────────────────────────────────────

def _page_regression(df: pd.DataFrame) -> None:
    st.markdown("## Regression & Insights")
    st.caption("Ridge regression (RidgeCV, 5-fold CV) with z-score standardized features — coefficients are directly comparable across predictors.")

    groups = get_feature_columns(df)
    all_features = (
        groups.get("sleep_lag", [])
        + groups.get("activity_lag", [])
        + groups.get("derived", [])
    )
    all_features = [f for f in all_features if f in df.columns and df[f].notna().sum() > 30]

    if len(all_features) < 2:
        st.warning(
            "Not enough biometric feature data. "
            "Wear your Oura ring for at least 30 days with overlapping glucose data."
        )
        return

    targets = [t for t in ["glucose_tir", "glucose_cv"] if t in df.columns]

    st.markdown("#### Predictor Features")
    feat_groups = {
        "Sleep (previous night)": [f for f in all_features if f.startswith("prev_night_")],
        "Activity & Readiness (previous day)": [f for f in all_features if f.startswith("prev_day_")],
        "Derived ratios": [f for f in all_features if not f.startswith(("prev_night_", "prev_day_"))],
    }

    selected: list[str] = []
    for group_name, cols in feat_groups.items():
        if cols:
            with st.expander(group_name, expanded=True):
                picked = st.multiselect(
                    "", cols,
                    default=cols[:4],
                    format_func=_label,
                    key=f"reg_{group_name}",
                )
                selected.extend(picked)

    if len(selected) < 1:
        st.info("Select at least one predictor feature.")
        return

    results = run_multi_target_regression(df, targets, selected)
    if not results:
        st.warning("Regressions could not be computed. Check for sufficient overlapping data.")
        return

    for target_name, res in results.items():
        st.markdown(f"### Target: {_label(target_name)}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{res.r_squared:.3f}")
        c2.metric("Adj. R²", f"{res.r_squared_adj:.3f}")
        c3.metric("Observations", res.n_observations)
        c4.metric("Ridge α", res.alpha)

        if not res.feature_importance.empty:
            imp = res.feature_importance.copy()
            imp["direction"] = imp["std_coefficient"].apply(
                lambda x: "Positive" if x >= 0 else "Negative"
            )
            imp["label"] = imp["feature"].apply(_label)

            fig = px.bar(
                imp.sort_values("abs_std_coefficient"),
                x="abs_std_coefficient",
                y="label",
                color="direction",
                color_discrete_map={"Positive": C["pos"], "Negative": C["neg"]},
                orientation="h",
                labels={"abs_std_coefficient": "Importance (|β|)", "label": ""},
            )
            fig.update_layout(
                height=max(260, 38 * len(imp)),
                margin=dict(t=20, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"reg_imp_{target_name}")

        with st.expander("Detailed coefficients"):
            disp = res.feature_importance[
                ["feature", "std_coefficient", "raw_coefficient"]
            ].copy()
            disp["feature"] = disp["feature"].apply(_label)
            disp.columns = ["Feature", "Std β", "Raw β"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown("---")


# ── Workout Analysis ──────────────────────────────────────────────────────────

def _page_workout_analysis(
    df: pd.DataFrame,
    raw_glucose: pd.DataFrame,
    workouts: pd.DataFrame,
) -> None:
    st.markdown("## Workout Analysis")
    st.caption("Glucose response before, during, and after exercise — by activity type.")

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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Workout Profiles", "Glucose Response Curve", "Delta Analysis", "Nadir & Timing",
    ])

    # ── Tab 1: Workout Profiles ──────────────────────────────────────────────

    with tab1:
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

    # ── Tab 2: Glucose Response Curve ────────────────────────────────────────

    with tab2:
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
            st.plotly_chart(fig, use_container_width=True, key="workout_response_curve")

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
                st.plotly_chart(fig2, use_container_width=True, key="workout_individual_traces")

    # ── Tab 3: Delta Analysis ────────────────────────────────────────────────

    with tab3:
        if not summary.empty:
            st.markdown("#### Glucose Change by Activity Type")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=summary["activity"],
                y=summary["avg_delta_during"],
                name="During workout",
                marker_color=C["chart1"],
                text=summary["avg_delta_during"].apply(lambda v: f"{v:+.0f}" if pd.notna(v) else ""),
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=summary["activity"],
                y=summary["avg_delta_post"],
                name="1h post workout",
                marker_color=C["chart2"],
                text=summary["avg_delta_post"].apply(lambda v: f"{v:+.0f}" if pd.notna(v) else ""),
                textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                yaxis_title="Avg glucose change (mg/dL)",
                height=380,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, key="workout_deltas_type")

        # Time of day comparison
        tod_agg = wg.groupby("time_of_day").agg(
            n=("activity", "size"),
            avg_delta_during=("delta_during", "mean"),
            avg_delta_post=("delta_post", "mean"),
        ).round(1).reset_index()
        if len(tod_agg) > 1:
            st.markdown("#### Glucose Change by Time of Day")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tod_agg["time_of_day"],
                y=tod_agg["avg_delta_during"],
                name="During workout",
                marker_color=C["chart1"],
                text=[f"n={n}" for n in tod_agg["n"]],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=tod_agg["time_of_day"],
                y=tod_agg["avg_delta_post"],
                name="1h post workout",
                marker_color=C["chart2"],
            ))
            fig.update_layout(
                barmode="group",
                yaxis_title="Avg glucose change (mg/dL)",
                height=350,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, key="workout_deltas_tod")

    # ── Tab 4: Nadir & Timing ────────────────────────────────────────────────

    with tab4:
        nadir_data = wg.dropna(subset=["nadir_post_120"])
        if nadir_data.empty:
            st.info("No nadir data available (need glucose readings in the 2h post-workout).")
        else:
            st.markdown("#### Post-Workout Glucose Nadir")
            st.caption("Lowest glucose reading within 2 hours after each workout ends.")

            c1, c2 = st.columns(2)
            c1.metric("Avg Nadir", f"{nadir_data['nadir_post_120'].mean():.0f} mg/dL")
            c2.metric("Avg Time to Nadir", f"{nadir_data['nadir_time_min'].mean():.0f} min")

            fig = go.Figure()
            colors = [C["chart1"], C["chart2"], C["chart3"], C["danger"], C["warning"]]
            for i, act in enumerate(nadir_data["activity"].unique()):
                act_d = nadir_data[nadir_data["activity"] == act]
                fig.add_trace(go.Scatter(
                    x=act_d["nadir_time_min"],
                    y=act_d["nadir_post_120"],
                    mode="markers+text",
                    name=act,
                    marker=dict(color=colors[i % len(colors)], size=10),
                    text=act_d["time_of_day"],
                    textposition="top center",
                    textfont=dict(size=10, color=C["text_sec"]),
                ))

            fig.add_hline(
                y=70, line_dash="dash", line_color=C["danger"],
                annotation_text="Hypo threshold (70)",
                annotation_font_color=C["danger"],
            )
            fig.update_layout(
                xaxis_title="Minutes after workout end",
                yaxis_title="Nadir glucose (mg/dL)",
                height=400,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, key="workout_nadir_scatter")

            # Per-type nadir summary table
            nadir_summary = nadir_data.groupby("activity").agg(
                n=("activity", "size"),
                avg_nadir=("nadir_post_120", "mean"),
                min_nadir=("nadir_post_120", "min"),
                avg_time=("nadir_time_min", "mean"),
            ).round(1).reset_index()
            nadir_summary.columns = ["Activity", "N", "Avg Nadir", "Lowest Nadir", "Avg Time (min)"]
            st.dataframe(nadir_summary, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Health Dashboard",
        page_icon="💜",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    if st.sidebar.button("Sync now"):
        _sync_all.clear()
        _load_events.clear()
        st.rerun()

    df, raw_glucose, workouts = _load_data()
    events = _load_events()
    filtered, page = _sidebar(df)

    if filtered.empty:
        st.warning("No data for the selected date range.")
        return

    if page == "Overview":
        _page_overview(filtered, raw_glucose)
    elif page == "Glucose Deep Dive":
        _page_glucose(filtered, raw_glucose, events)
    elif page == "Lifestyle Factors":
        _page_lifestyle(filtered)
    elif page == "Correlation Explorer":
        _page_correlations(filtered)
    elif page == "Workout Analysis":
        _page_workout_analysis(filtered, raw_glucose, workouts)
    elif page == "Regression & Insights":
        _page_regression(filtered)


if __name__ == "__main__":
    main()

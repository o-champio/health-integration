"""Centralized theme: palette, fonts, Plotly template, CSS, and helpers.

All four pages and the insight cards import the palette `C` from here.
Never re-declare the palette in a page module.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ── Palette ───────────────────────────────────────────────────────────────────

C: dict[str, str] = {
    # Surfaces / chrome
    "bg":          "#1A1612",
    "card":        "#221C16",
    "surface":     "#2A2520",
    "border":      "#3A332C",
    # Text
    "text":        "#F5EFE6",
    "text_sec":    "#C8B8A2",
    "text_muted":  "#A89886",
    # Brand
    "primary":     "#E8855C",   # terracotta
    "secondary":   "#E879F9",   # magenta
    # Semantic (glucose convention)
    "glucose":     "#34D399",   # in-range mint
    "warning":     "#D4A574",   # above-range muted gold
    "danger":      "#EF4444",   # below-range red
    "chart_cool":  "#38BDF8",   # activity sky
    # Back-compat aliases used by existing chart code
    "success":     "#34D399",
    "accent":      "#E879F9",
    "accent_soft": "#F5B6FA",
    "sleep":       "#E879F9",
    "activity":    "#38BDF8",
    "chart1":      "#E8855C",
    "chart2":      "#38BDF8",
    "chart3":      "#34D399",
    "pos":         "#34D399",
    "neg":         "#EF4444",
    "neutral":     "#A89886",
}


# ── Plotly modebar config ─────────────────────────────────────────────────────

PLOTLY_CONFIG: dict = {"displayModeBar": False}


# ── Delta color semantics ─────────────────────────────────────────────────────

_HIGHER_IS_BETTER = {
    "glucose_tir", "sleep_score", "readiness_score", "activity_score",
    "session_avg_hrv", "prev_night_sleep_score", "prev_day_readiness_score",
    "prev_day_activity_score", "prev_night_hrv",
}
_LOWER_IS_BETTER = {
    "glucose_mean", "glucose_cv", "glucose_tbr", "glucose_tar", "glucose_std",
}


def delta_color_for(metric: str) -> str:
    """Return the `delta_color` value for `st.metric` based on the metric name.

    Returns "normal" (green-up) when higher is better, "inverse" (red-up) when
    lower is better, and "off" for anything not recognized.
    """
    if metric in _HIGHER_IS_BETTER:
        return "normal"
    if metric in _LOWER_IS_BETTER:
        return "inverse"
    return "off"


# ── Plotly template ───────────────────────────────────────────────────────────

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
        font=dict(color=C["text"], family="Manrope, system-ui, sans-serif", size=13),
        title_font=dict(color=C["text"], size=15, family="Manrope, system-ui, sans-serif"),
        xaxis=_AXIS_STYLE,
        yaxis=_AXIS_STYLE,
        legend=dict(
            bgcolor=C["card"],
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text_sec"], size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        colorway=[C["chart1"], C["chart2"], C["chart3"], C["warning"], C["danger"], C["secondary"]],
        margin=dict(t=44, b=44, l=52, r=44),
        hoverlabel=dict(
            bgcolor=C["surface"],
            bordercolor=C["border"],
            font=dict(color=C["text"], size=13),
        ),
    )
)


# ── CSS (desktop + mobile @media block) ───────────────────────────────────────

_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&display=swap');

  /* ── Base ──────────────────────────────────────────────── */
  html, body, .stApp, [class*="css"] {{
    font-family: 'Manrope', system-ui, sans-serif !important;
  }}
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
  [data-baseweb="select"] [data-baseweb="tag"] {{ background: {C['surface']} !important; }}
  [data-baseweb="select"] span {{ color: {C['text']} !important; }}
  [data-baseweb="menu"] li {{ color: {C['text']} !important; background: {C['card']} !important; }}
  [data-baseweb="menu"] li:hover {{ background: {C['surface']} !important; }}
  [data-testid="stDateInput"] input {{ color: {C['text']} !important; }}

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
    background: rgba(212,165,116,0.14);
    border: 1px solid {C['warning']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
  .insight-card {{
    background: rgba(232,133,92,0.12);
    border: 1px solid {C['primary']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}
  .success-card {{
    background: rgba(52,211,153,0.12);
    border: 1px solid {C['success']};
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: {C['text']} !important;
  }}

  /* ── Mobile (<768px) ───────────────────────────────────── */
  @media (max-width: 768px) {{
    .main .block-container {{ padding: 1rem 0.75rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
    [data-testid="stMetric"] {{ padding: 12px 14px; }}
    .stTabs [data-baseweb="tab"] {{ padding: 6px 10px; font-size: 0.8rem; }}
    h1 {{ font-size: 1.5rem !important; }}
    h2 {{ font-size: 1.2rem !important; }}
    h3 {{ font-size: 1.05rem !important; }}
  }}
</style>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def apply_theme() -> None:
    """Install the Plotly template default and inject the CSS.

    Idempotent: safe to call from `main()` on every rerun.
    """
    pio.templates.default = "health"
    st.markdown(_CSS, unsafe_allow_html=True)

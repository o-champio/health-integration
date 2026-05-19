"""Health Dashboard -- T1D-focused Streamlit application.

Launch:
    streamlit run app/main.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.processing.features import build_analysis_df
from src.processing.pipeline import sync_all
from app._shared import (
    _sidebar_filters,
    _load_events,
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


# _CAT_COLS, _LABEL_FIXES, _avail, _label — imported from app._shared


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


# ── Page nav callables ────────────────────────────────────────────────────────────────

def _get_shared_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, None]:
    """Load all data sources; each _nav_* applies _sidebar_filters itself."""
    df, raw_glucose, workouts = _load_data()
    events = _load_events()
    return raw_glucose, df, workouts, events, None


def _nav_insights() -> None:
    raw_glucose, df, workouts, events, _wg = _get_shared_state()
    df = _sidebar_filters(df)
    from app.pages import insights
    insights.render(df, raw_glucose)


def _nav_glucose() -> None:
    raw_glucose, df, workouts, events, _wg = _get_shared_state()
    df = _sidebar_filters(df)
    from app.pages import glucose
    glucose.render(df, raw_glucose, events)


def _nav_lifestyle() -> None:
    _raw, df, _wk, _ev, _wg = _get_shared_state()
    df = _sidebar_filters(df)
    from app.pages import lifestyle
    lifestyle.render(df)


def _nav_workouts() -> None:
    raw_glucose, df, workouts, _ev, _wg = _get_shared_state()
    df = _sidebar_filters(df)
    from app.pages import workouts as workouts_page
    workouts_page.render(df, raw_glucose, workouts)


# ── Main ────────────────────────────────────────────────────────────────────────────

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

    pages = [
        st.Page(_nav_insights,  title="Insights",           default=True),
        st.Page(_nav_glucose,   title="Glucose Deep Dive"),
        st.Page(_nav_lifestyle, title="Lifestyle Factors"),
        st.Page(_nav_workouts,  title="Workout Analysis"),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()

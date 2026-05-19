"""Health Dashboard -- T1D-focused Streamlit application.

Launch:
    streamlit run app/main.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
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
from app._theme import apply_theme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



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
    apply_theme()

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

"""Health Dashboard -- Streamlit application.

Launch:
    streamlit run app/main.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path so imports resolve
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.analysis import (
    RegressionResult,
    dual_correlation,
    run_multi_target_regression,
)
from src.processing.features import (
    build_analysis_df,
    get_feature_columns,
    get_regression_ready,
)
from src.processing.pipeline import build_daily_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Data loading (cached, auto-syncs every 12 h) -----------------------------

@st.cache_data(ttl=43200, show_spinner="Syncing data from Oura API and LibreLink...")
def _sync_and_load() -> pd.DataFrame:
    """Run the full pipeline and return the analysis-ready DataFrame.

    Cached for 12 hours (43 200 s). On first load (or cache expiry) it
    fetches new data from Oura incrementally and reloads Libre CSVs.
    """
    daily = build_daily_dataset(incremental=True)
    analysis = build_analysis_df(daily)
    return analysis


def _load_data() -> pd.DataFrame:
    """Load data, showing a user-friendly error if something fails."""
    try:
        return _sync_and_load()
    except FileNotFoundError as exc:
        st.error(f"Data not found: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        log.exception("Pipeline failed")
        st.stop()


# -- Sidebar -------------------------------------------------------------------

def _sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Render sidebar controls and return (filtered_df, start_str, end_str)."""
    st.sidebar.header("Filters")

    dates = pd.to_datetime(df["date"])
    min_date = dates.min().date()
    max_date = dates.max().date()

    default_start = max(min_date, max_date - timedelta(days=90))
    start, end = st.sidebar.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    filtered = df[
        (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Days in range", len(filtered))

    # Quick summary metrics
    if "glucose_tir" in filtered.columns:
        avg_tir = filtered["glucose_tir"].mean()
        if pd.notna(avg_tir):
            st.sidebar.metric("Avg TIR", f"{avg_tir:.1%}")
    if "glucose_cv" in filtered.columns:
        avg_cv = filtered["glucose_cv"].mean()
        if pd.notna(avg_cv):
            st.sidebar.metric("Avg Glucose CV", f"{avg_cv:.3f}")
    if "prev_night_hrv" in filtered.columns:
        avg_hrv = filtered["prev_night_hrv"].mean()
        if pd.notna(avg_hrv):
            st.sidebar.metric("Avg Nightly HRV", f"{avg_hrv:.1f} ms")

    return filtered, str(start), str(end)


# -- Tab 1: Trend Analysis ----------------------------------------------------

def _tab_trends(df: pd.DataFrame) -> None:
    """Dual-axis time series: glucose metrics overlaid with biometric data."""
    st.subheader("Glucose and Biometric Trends")

    col1, col2 = st.columns(2)
    glucose_options = [c for c in df.columns if c.startswith("glucose_") and "7d" not in c]
    biometric_options = [
        c for c in df.columns
        if c.startswith(("prev_night_", "prev_day_", "sleep_score", "readiness_score",
                         "activity_score", "session_"))
        and df[c].notna().sum() > 10
    ]

    if not glucose_options:
        st.info("No glucose data available for the selected range.")
        return

    with col1:
        y_glucose = st.selectbox("Glucose metric (left axis)", glucose_options,
                                 index=glucose_options.index("glucose_tir")
                                 if "glucose_tir" in glucose_options else 0)
    with col2:
        if biometric_options:
            default_bio = "prev_night_hrv" if "prev_night_hrv" in biometric_options else 0
            y_bio = st.selectbox("Biometric metric (right axis)", biometric_options,
                                 index=biometric_options.index(default_bio)
                                 if isinstance(default_bio, str) else 0)
        else:
            y_bio = None

    smoothing = st.slider("Smoothing window (days)", min_value=1, max_value=30, value=7)

    plot_df = df[["date"]].copy()
    plot_df[y_glucose] = df[y_glucose].rolling(smoothing, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df[y_glucose],
        name=_label(y_glucose), yaxis="y1",
        line=dict(color="#1f77b4", width=2),
    ))

    if y_bio is not None and y_bio in df.columns:
        plot_df[y_bio] = df[y_bio].rolling(smoothing, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df[y_bio],
            name=_label(y_bio), yaxis="y2",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
        ))

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title=dict(text=_label(y_glucose), font=dict(color="#1f77b4")),
                   tickfont=dict(color="#1f77b4")),
        yaxis2=dict(title=dict(text=_label(y_bio) if y_bio else "", font=dict(color="#ff7f0e")),
                    tickfont=dict(color="#ff7f0e"),
                    overlaying="y", side="right") if y_bio else {},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling TIR bands
    if "glucose_tir" in df.columns and len(df) >= 7:
        st.subheader("7-Day Rolling Time in Range")
        tir_df = df[["date", "glucose_tir"]].dropna().copy()
        tir_df["tir_7d"] = tir_df["glucose_tir"].rolling(7, min_periods=3).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=tir_df["date"], y=tir_df["glucose_tir"],
            name="Daily TIR", mode="markers",
            marker=dict(size=4, color="#aec7e8", opacity=0.5),
        ))
        fig2.add_trace(go.Scatter(
            x=tir_df["date"], y=tir_df["tir_7d"],
            name="7-day rolling", line=dict(color="#1f77b4", width=2.5),
        ))
        fig2.add_hline(y=0.70, line_dash="dash", line_color="green",
                       annotation_text="70% target")
        fig2.update_layout(
            yaxis=dict(title="Time in Range", tickformat=".0%"),
            height=350, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)


# -- Tab 2: Correlation Explorer -----------------------------------------------

def _tab_correlations(df: pd.DataFrame) -> None:
    """Interactive scatter plot + correlation heatmap."""
    st.subheader("Scatter Plot Explorer")

    groups = get_feature_columns(df)
    all_numeric = [c for c in df.select_dtypes(include="number").columns
                   if df[c].notna().sum() > 20]

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X axis", all_numeric,
                             index=all_numeric.index("prev_night_hrv")
                             if "prev_night_hrv" in all_numeric else 0)
    with col2:
        default_y = "glucose_mean" if "glucose_mean" in all_numeric else 0
        y_col = st.selectbox("Y axis", all_numeric,
                             index=all_numeric.index(default_y)
                             if isinstance(default_y, str) else 0)

    scatter_df = df[[x_col, y_col]].dropna()
    if scatter_df.empty:
        st.warning("No overlapping data for the selected columns.")
        return

    fig = px.scatter(
        scatter_df, x=x_col, y=y_col,
        trendline="ols", trendline_color_override="#ff7f0e",
        labels={x_col: _label(x_col), y_col: _label(y_col)},
        opacity=0.6,
    )
    fig.update_layout(height=450, margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Stats for the pair
    pearson_r = scatter_df[x_col].corr(scatter_df[y_col], method="pearson")
    spearman_r = scatter_df[x_col].corr(scatter_df[y_col], method="spearman")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson r", f"{pearson_r:.3f}")
    c2.metric("Spearman rho", f"{spearman_r:.3f}")
    c3.metric("Observations", len(scatter_df))

    # Full heatmap
    st.markdown("---")
    st.subheader("Correlation Matrix (Spearman)")

    glucose_cols = groups.get("glucose", [])
    bio_cols = groups.get("sleep_lag", []) + groups.get("activity_lag", []) + groups.get("derived", [])
    bio_cols = [c for c in bio_cols if c in all_numeric]

    if glucose_cols and bio_cols:
        corr_matrices = dual_correlation(df, glucose_cols, bio_cols)
        method = st.radio("Method", ["spearman", "pearson"], horizontal=True)
        corr = corr_matrices[method]

        if not corr.empty:
            fig_hm = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                labels=dict(x="Biometric Feature", y="Glucose Metric", color="Correlation"),
                aspect="auto",
            )
            fig_hm.update_layout(height=max(350, 50 * len(corr)), margin=dict(t=30, b=30))
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Not enough overlapping biometric and glucose data for a correlation matrix.")


# -- Tab 3: Model Insights ----------------------------------------------------

def _tab_models(df: pd.DataFrame) -> None:
    """Regression results and feature importance."""
    st.subheader("Regression Analysis")

    groups = get_feature_columns(df)
    available_features = (
        groups.get("sleep_lag", [])
        + groups.get("activity_lag", [])
        + groups.get("derived", [])
    )
    available_features = [f for f in available_features if f in df.columns
                          and df[f].notna().sum() > 30]

    if len(available_features) < 2:
        st.warning(
            "Not enough biometric feature data to run regressions. "
            "Wear your Oura ring for at least 30 days with overlapping glucose data."
        )
        return

    targets = ["glucose_tir", "glucose_cv"]
    targets = [t for t in targets if t in df.columns]

    selected_features = st.multiselect(
        "Predictor features",
        available_features,
        default=available_features[:6],
    )

    if len(selected_features) < 1:
        st.info("Select at least one predictor feature.")
        return

    results = run_multi_target_regression(df, targets, selected_features)

    if not results:
        st.warning("Regressions could not be computed. Check for sufficient overlapping data.")
        return

    for target_name, res in results.items():
        st.markdown(f"### Target: {_label(target_name)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("R-squared", f"{res.r_squared:.3f}")
        c2.metric("Adj. R-squared", f"{res.r_squared_adj:.3f}")
        c3.metric("Observations", res.n_observations)

        # Feature importance bar chart
        if not res.feature_importance.empty:
            imp = res.feature_importance.copy()
            imp["direction"] = imp["std_coefficient"].apply(
                lambda x: "positive" if x >= 0 else "negative"
            )
            imp["feature_label"] = imp["feature"].apply(_label)

            fig = px.bar(
                imp.sort_values("abs_std_coefficient"),
                x="abs_std_coefficient", y="feature_label",
                color="direction",
                color_discrete_map={"positive": "#2ca02c", "negative": "#d62728"},
                orientation="h",
                labels={"abs_std_coefficient": "Absolute Standardized Coefficient",
                        "feature_label": ""},
            )
            fig.update_layout(
                showlegend=True,
                height=max(300, 40 * len(imp)),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Coefficients table
        with st.expander("Detailed coefficients"):
            display_df = res.feature_importance[
                ["feature", "coefficient", "std_coefficient", "p_value", "significant"]
            ].copy()
            display_df["feature"] = display_df["feature"].apply(_label)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Full statsmodels summary
        with st.expander("Full OLS summary"):
            st.code(res.summary_text, language="text")

        st.markdown("---")


# -- Helpers -------------------------------------------------------------------

_LABEL_MAP = {
    "Tir": "TIR",
    "Tbr": "TBR",
    "Tar": "TAR",
    " Cv": " CV",
    "Gmi": "GMI",
    "Hrv": "HRV",
    " Hr": " HR",
    "Ols": "OLS",
}


def _label(col_name: str) -> str:
    """Convert column names to readable labels."""
    text = (
        col_name
        .replace("glucose_", "Glucose ")
        .replace("prev_night_", "Prev Night ")
        .replace("prev_day_", "Prev Day ")
        .replace("session_", "Sleep ")
        .replace("sleep_", "Sleep ")
        .replace("readiness_", "Readiness ")
        .replace("activity_", "Activity ")
        .replace("stress_", "Stress ")
        .replace("contributors.", "")
        .replace("_", " ")
        .title()
    )
    for wrong, right in _LABEL_MAP.items():
        text = text.replace(wrong, right)
    return text


# -- Main app ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Health Data Dashboard",
        page_icon=None,
        layout="wide",
    )
    st.title("Health Data Dashboard")
    st.caption("Oura Ring biometrics + Continuous Glucose Monitor analysis")

    df = _load_data()
    filtered, start, end = _sidebar(df)

    if filtered.empty:
        st.warning("No data for the selected date range.")
        return

    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Correlation Explorer", "Predictive Insights"])

    with tab1:
        _tab_trends(filtered)
    with tab2:
        _tab_correlations(filtered)
    with tab3:
        _tab_models(filtered)


if __name__ == "__main__":
    main()

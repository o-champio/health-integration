"""Statistical analysis: correlation matrices and Ridge regression models.

All functions accept the analysis-ready DataFrame from
src.processing.features.build_analysis_df().

Standardization strategy
------------------------
Features (X) are z-score standardized before regression so that coefficients
are directly comparable regardless of original scale (HRV in ms, sleep in
minutes, step counts in thousands, etc.). The target (y) is kept in its
original units so the intercept and raw coefficients remain interpretable.

Consequence: model.coef_ are already standardized beta weights. Raw
(unstandardized) coefficients are back-computed as beta / std(X) and stored
separately for reporting purposes.

Ridge vs OLS
------------
RidgeCV is used instead of OLS because the dataset is small (~100-200 rows)
and Oura biometric features are highly collinear (sleep score, HRV, readiness
all move together). Ridge L2 regularization stabilizes coefficients in this
setting. The optimal alpha is selected automatically via 5-fold cross-validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -- Correlation ---------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    glucose_cols: list[str],
    feature_cols: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute a correlation matrix between glucose metrics and biometric features.

    Parameters
    ----------
    df : analysis DataFrame (NaNs handled via pairwise deletion)
    glucose_cols : glucose target columns (rows in the output matrix)
    feature_cols : biometric feature columns (columns in the output matrix)
    method : 'pearson' or 'spearman'

    Returns
    -------
    DataFrame with glucose_cols as index, feature_cols as columns, values = correlation.
    """
    all_cols = [c for c in glucose_cols + feature_cols if c in df.columns]
    if len(all_cols) < 2:
        return pd.DataFrame()

    corr_full = df[all_cols].corr(method=method)

    g_avail = [c for c in glucose_cols if c in corr_full.index]
    f_avail = [c for c in feature_cols if c in corr_full.columns]

    return corr_full.loc[g_avail, f_avail].round(3)


def dual_correlation(
    df: pd.DataFrame,
    glucose_cols: list[str],
    feature_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Return both Pearson and Spearman correlation matrices."""
    return {
        "pearson": correlation_matrix(df, glucose_cols, feature_cols, method="pearson"),
        "spearman": correlation_matrix(df, glucose_cols, feature_cols, method="spearman"),
    }


# -- Regression ----------------------------------------------------------------

@dataclass
class RegressionResult:
    """Container for Ridge regression model outputs.

    coefficients        : standardized beta weights (X was z-scored before fitting)
    raw_coefficients    : back-transformed to original feature units (beta / std(X))
    intercept           : intercept in original target units (y is never standardized)
    alpha               : regularization strength chosen by RidgeCV
    """
    target: str
    features: list[str]
    coefficients: dict[str, float]        # standardized (beta weights)
    raw_coefficients: dict[str, float]    # original units
    intercept: float
    r_squared: float
    r_squared_adj: float
    n_observations: int
    alpha: float
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> RegressionResult:
    """Run Ridge regression on z-scored features and return structured results.

    Features are z-score standardized (zero mean, unit variance) before
    fitting so coefficients are directly comparable across different scales.
    The target is kept in its original units. Alpha is selected via 5-fold
    cross-validation from a log-spaced grid.
    """
    cols = [target] + features
    clean = df[cols].dropna()

    if len(clean) < len(features) + 10:
        raise ValueError(
            f"Too few observations ({len(clean)}) for {len(features)} features. "
            f"Need at least {len(features) + 10}."
        )

    X_raw = clean[features]
    y = clean[target]

    # Z-score standardize X; drop any feature with zero variance (constant column)
    X_mean = X_raw.mean()
    X_std = X_raw.std()
    zero_var = X_std[X_std == 0].index.tolist()
    if zero_var:
        log.warning("Dropping zero-variance features: %s", zero_var)
        features = [f for f in features if f not in zero_var]
        X_raw = X_raw[features]
        X_mean = X_mean[features]
        X_std = X_std[features]

    X_scaled = (X_raw - X_mean) / X_std

    return _ridge_sklearn(X_scaled, X_std, y, target, features)


def _ridge_sklearn(
    X_scaled: pd.DataFrame,
    X_std: pd.Series,
    y: pd.Series,
    target: str,
    features: list[str],
) -> RegressionResult:
    """Ridge regression on z-scored X with CV alpha selection."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_scaled, y)

    n = len(y)
    k = len(features)
    r2 = float(r2_score(y, model.predict(X_scaled)))
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    std_coefs = {f: round(float(coef), 6) for f, coef in zip(features, model.coef_)}
    raw_coefs = {f: round(std_coefs[f] / X_std[f], 6) for f in features}

    importance = pd.DataFrame({
        "feature": features,
        "std_coefficient": [std_coefs[f] for f in features],
        "abs_std_coefficient": [round(abs(std_coefs[f]), 4) for f in features],
        "raw_coefficient": [raw_coefs[f] for f in features],
    }).sort_values("abs_std_coefficient", ascending=False).reset_index(drop=True)

    return RegressionResult(
        target=target,
        features=features,
        coefficients=std_coefs,
        raw_coefficients=raw_coefs,
        intercept=round(float(model.intercept_), 6),
        r_squared=round(r2, 4),
        r_squared_adj=round(r2_adj, 4),
        n_observations=n,
        alpha=round(float(model.alpha_), 4),
        feature_importance=importance,
    )


def run_multi_target_regression(
    df: pd.DataFrame,
    targets: list[str],
    features: list[str],
) -> dict[str, RegressionResult]:
    """Run regression for multiple glucose targets, returning a dict of results."""
    results = {}
    for t in targets:
        if t not in df.columns:
            log.warning("Target '%s' not in DataFrame, skipping.", t)
            continue
        try:
            results[t] = run_regression(df, t, features)
            log.info(
                "Ridge %s: R2=%.3f, R2_adj=%.3f, alpha=%.4f, n=%d",
                t, results[t].r_squared, results[t].r_squared_adj,
                results[t].alpha, results[t].n_observations,
            )
        except ValueError as exc:
            log.warning("Skipping target '%s': %s", t, exc)
    return results

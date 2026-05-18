"""Shared helpers for insight cards.

All insight modules import from here:
- ``filter_rest_mode``: drop ``in_rest_mode == True`` rows before any analysis.
- ``monitored_caveat``: build the standard caveat dict used by literature-monitored cards.
"""
from __future__ import annotations

from typing import TypedDict

import pandas as pd


class MonitoredCaveat(TypedDict):
    n: int
    rho: float
    p: float
    monitored: bool
    significant: bool


def filter_rest_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ``in_rest_mode`` is truthy. Pass-through if column absent."""
    if "in_rest_mode" not in df.columns:
        return df
    keep = ~df["in_rest_mode"].astype("boolean").fillna(False).astype(bool)
    return df.loc[keep].copy()


def monitored_caveat(n: int, rho: float, p: float) -> MonitoredCaveat:
    """Standard caveat dict for literature-monitored correlation cards.

    ``significant`` follows the conventional p < 0.05 threshold; cards use it
    to choose between "tracking" and "showing a signal" copy.
    """
    return {
        "n": n,
        "rho": float(rho),
        "p": float(p),
        "monitored": True,
        "significant": bool(p < 0.05),
    }

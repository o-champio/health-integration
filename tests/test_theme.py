"""Tests for the centralized theme module."""
from __future__ import annotations

import pytest


REQUIRED_PALETTE_KEYS = [
    "bg", "card", "surface", "border",
    "text", "text_sec", "text_muted",
    "primary", "secondary",
    "glucose", "warning", "danger", "chart_cool",
    # Legacy/back-compat keys still referenced by existing chart code:
    "success", "accent", "accent_soft", "sleep", "activity",
    "chart1", "chart2", "chart3", "pos", "neg", "neutral",
]


def test_palette_has_required_keys():
    from app._theme import C
    missing = [k for k in REQUIRED_PALETTE_KEYS if k not in C]
    assert not missing, f"Palette missing keys: {missing}"


def test_palette_values_are_hex_strings():
    from app._theme import C
    for key, val in C.items():
        assert isinstance(val, str) and val.startswith("#"), \
            f"C[{key!r}] = {val!r} is not a hex color string"


def test_plotly_config_hides_modebar():
    from app._theme import PLOTLY_CONFIG
    assert PLOTLY_CONFIG == {"displayModeBar": False}


def test_delta_color_for_higher_is_better():
    from app._theme import delta_color_for
    for metric in ("glucose_tir", "sleep_score", "readiness_score",
                   "activity_score", "session_avg_hrv"):
        assert delta_color_for(metric) == "normal", metric


def test_delta_color_for_lower_is_better():
    from app._theme import delta_color_for
    for metric in ("glucose_mean", "glucose_cv", "glucose_tbr", "glucose_tar"):
        assert delta_color_for(metric) == "inverse", metric


def test_delta_color_for_unknown_metric_defaults_to_off():
    from app._theme import delta_color_for
    assert delta_color_for("does_not_exist") == "off"

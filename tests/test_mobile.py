"""Tests for viewport detection helpers."""
from __future__ import annotations


def test_mobile_from_width_below_threshold():
    from app._mobile import _mobile_from_width
    assert _mobile_from_width(360) is True
    assert _mobile_from_width(430) is True
    assert _mobile_from_width(767) is True


def test_mobile_from_width_at_or_above_threshold():
    from app._mobile import _mobile_from_width
    assert _mobile_from_width(768) is False
    assert _mobile_from_width(1024) is False
    assert _mobile_from_width(1920) is False


def test_mobile_from_width_none_defaults_to_desktop():
    """When the JS round-trip hasn't returned yet, default to desktop layout."""
    from app._mobile import _mobile_from_width
    assert _mobile_from_width(None) is False


def test_mobile_from_width_zero_or_negative_defaults_to_desktop():
    from app._mobile import _mobile_from_width
    assert _mobile_from_width(0) is False
    assert _mobile_from_width(-1) is False

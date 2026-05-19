"""Viewport detection for mobile-vs-desktop layout branching.

`is_mobile()` is called once per page render. The actual JS round-trip
(`streamlit_js_eval`) only fires the first time per session; subsequent calls
read the cached value from `st.session_state`.
"""
from __future__ import annotations

import streamlit as st

MOBILE_BREAKPOINT_PX = 768
_SESSION_KEY = "_viewport_w"


def _mobile_from_width(width: int | None) -> bool:
    """Pure function: True iff width is a positive number below the breakpoint."""
    if width is None or width <= 0:
        return False
    return width < MOBILE_BREAKPOINT_PX


def viewport_width() -> int | None:
    """Return the cached viewport width, querying the browser if not yet cached.

    Returns None on the first call of a session (the JS eval is async-ish via
    Streamlit reruns; the value will be present on the next rerun).
    """
    cached = st.session_state.get(_SESSION_KEY)
    if cached is not None:
        return cached
    try:
        from streamlit_js_eval import streamlit_js_eval
    except ImportError:
        return None
    w = streamlit_js_eval(js_expressions="window.innerWidth", key="vw")
    if w is not None:
        try:
            w_int = int(w)
        except (TypeError, ValueError):
            return None
        st.session_state[_SESSION_KEY] = w_int
        return w_int
    return None


def is_mobile() -> bool:
    """True when the viewport is narrower than the mobile breakpoint."""
    return _mobile_from_width(viewport_width())

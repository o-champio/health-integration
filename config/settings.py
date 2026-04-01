"""Central configuration: paths, API endpoints, scopes, and credentials.

Credentials are loaded from environment variables first, falling back
to config/credentials.py for local development.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TOKEN_DIR = PROJECT_ROOT / "auth" / "tokens"
TOKEN_FILE = str(TOKEN_DIR / "oura_token.json")

# ── Credentials ──────────────────────────────────────────────────────────────
# Prefer env vars; fall back to config/credentials.py for local dev.

def _load_credentials() -> tuple[str, str]:
    client_id = os.environ.get("OURA_CLIENT_ID")
    client_secret = os.environ.get("OURA_CLIENT_SECRET")

    if client_id and client_secret:
        return client_id, client_secret

    try:
        from config import credentials as _c
        return _c.OURA_CLIENT_ID, _c.OURA_CLIENT_SECRET
    except (ImportError, AttributeError):
        log.warning(
            "Oura credentials not found. Set OURA_CLIENT_ID / OURA_CLIENT_SECRET "
            "env vars or create config/credentials.py from credentials.example.py."
        )
        return "", ""


OURA_CLIENT_ID, OURA_CLIENT_SECRET = _load_credentials()


def _load_dexcom_credentials() -> tuple[str, str, bool]:
    client_id = os.environ.get("DEXCOM_CLIENT_ID")
    client_secret = os.environ.get("DEXCOM_CLIENT_SECRET")
    sandbox = os.environ.get("DEXCOM_SANDBOX", "").lower() in ("1", "true", "yes")

    if client_id and client_secret:
        return client_id, client_secret, sandbox

    try:
        from config import credentials as _c
        return (
            getattr(_c, "DEXCOM_CLIENT_ID", ""),
            getattr(_c, "DEXCOM_CLIENT_SECRET", ""),
            getattr(_c, "DEXCOM_SANDBOX", False),
        )
    except ImportError:
        log.warning(
            "Dexcom credentials not found. Set DEXCOM_CLIENT_ID / DEXCOM_CLIENT_SECRET "
            "env vars or add them to config/credentials.py."
        )
        return "", "", False


DEXCOM_CLIENT_ID, DEXCOM_CLIENT_SECRET, DEXCOM_SANDBOX = _load_dexcom_credentials()

# ── Oura API ─────────────────────────────────────────────────────────────────

OURA_REDIRECT_URL = "http://localhost:8080"
AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL = "https://api.ouraring.com/oauth/token"
BASE_URL = "https://api.ouraring.com/v2/usercollection/"

SCOPES = [
    "daily",      # Readiness and sleep summaries
    "heartrate",  # HRV and stress data
    "personal",   # Profile data
    "workout",    # Exercise impact on glycemia
    "tag",        # Insulin/carb annotations
]

# ── Dexcom API ───────────────────────────────────────────────────────────────

DEXCOM_TOKEN_FILE = str(TOKEN_DIR / "dexcom_token.json")
DEXCOM_REDIRECT_URL = "http://localhost:8080"

_DEXCOM_BASE = "https://sandbox-api.dexcom.com" if DEXCOM_SANDBOX else "https://api.dexcom.com"
DEXCOM_AUTH_URL = f"{_DEXCOM_BASE}/v2/oauth2/login"
DEXCOM_TOKEN_URL = f"{_DEXCOM_BASE}/v2/oauth2/token"
DEXCOM_BASE_URL = f"{_DEXCOM_BASE}/v3/"
DEXCOM_SCOPES = ["offline_access"]

# ── Glucose thresholds (mg/dL) ───────────────────────────────────────────────

GLUCOSE_LOW = 70
GLUCOSE_HIGH = 180

# ── Timezone ─────────────────────────────────────────────────────────────────

LOCAL_TIMEZONE = "America/Sao_Paulo"

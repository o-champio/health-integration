"""Interactive OAuth2 flow for Oura and Dexcom APIs.

Usage from project root:
    python -m auth.oauth           # Oura (default)
    python -m auth.oauth oura      # Oura explicit
    python -m auth.oauth dexcom    # Dexcom
"""
from __future__ import annotations

import json
import logging
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from config import settings as cfg

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

log = logging.getLogger(__name__)


def _extract_code(redirect_url: str) -> str:
    """Pull the 'code' query parameter from the redirect URL."""
    params = parse_qs(urlparse(redirect_url).query)
    codes = params.get("code")
    if not codes:
        raise ValueError(f"No 'code' parameter found in: {redirect_url}")
    return codes[0]


# ── Oura ─────────────────────────────────────────────────────────────────────

def _exchange_code_oura(code: str) -> dict:
    resp = requests.post(
        cfg.TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.OURA_REDIRECT_URI,
            "client_id": cfg.OURA_CLIENT_ID,
            "client_secret": cfg.OURA_CLIENT_SECRET,
        },
    )
    if resp.status_code != 200:
        log.error("Oura token exchange failed: %s", resp.text)
    resp.raise_for_status()
    return resp.json()


def run_oura_oauth_flow() -> dict:
    """Run the interactive browser-based OAuth2 code-grant flow for Oura."""
    from requests_oauthlib import OAuth2Session

    oura = OAuth2Session(
        cfg.OURA_CLIENT_ID,
        scope=cfg.SCOPES,
        redirect_uri=cfg.OURA_REDIRECT_URI,
    )
    authorization_url, _ = oura.authorization_url(cfg.AUTH_URL)

    print("\n=== Oura OAuth Setup ===")
    print(f"\n1. Opening browser (or copy URL manually):\n   {authorization_url}\n")
    try:
        webbrowser.open(authorization_url)
    except Exception:
        pass

    redirect_response = input("2. Paste the full redirect URL here: ").strip()

    code = _extract_code(redirect_response)
    token = _exchange_code_oura(code)

    Path(cfg.TOKEN_DIR).mkdir(parents=True, exist_ok=True)
    with open(cfg.TOKEN_FILE, "w") as f:
        json.dump(token, f)
    log.info("Token saved to %s", cfg.TOKEN_FILE)

    # Quick connection test
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    resp = requests.get(f"{cfg.BASE_URL}personal_info", headers=headers)
    if resp.ok:
        info = resp.json()
        print(f"Connected!  User: {info.get('id')}  Age: {info.get('age')}")
    else:
        print(f"Warning: {resp.status_code} {resp.text}")

    return token


# ── Dexcom ────────────────────────────────────────────────────────────────────

def _exchange_code_dexcom(code: str) -> dict:
    resp = requests.post(
        cfg.DEXCOM_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.DEXCOM_REDIRECT_URI,
            "client_id": cfg.DEXCOM_CLIENT_ID,
            "client_secret": cfg.DEXCOM_CLIENT_SECRET,
        },
    )
    if resp.status_code != 200:
        log.error("Dexcom token exchange failed: %s", resp.text)
    resp.raise_for_status()
    return resp.json()


def run_dexcom_oauth_flow() -> dict:
    """Run the interactive browser-based OAuth2 code-grant flow for Dexcom."""
    from requests_oauthlib import OAuth2Session

    if not cfg.DEXCOM_CLIENT_ID:
        raise RuntimeError(
            "Dexcom credentials not configured. Add DEXCOM_CLIENT_ID and "
            "DEXCOM_CLIENT_SECRET to config/credentials.py or set env vars."
        )

    env_label = "SANDBOX" if cfg.DEXCOM_SANDBOX else "PRODUCTION"
    print(f"\n=== Dexcom OAuth Setup ({env_label}) ===")

    dexcom = OAuth2Session(
        cfg.DEXCOM_CLIENT_ID,
        scope=cfg.DEXCOM_SCOPES,
        redirect_uri=cfg.DEXCOM_REDIRECT_URI,
    )
    authorization_url, _ = dexcom.authorization_url(cfg.DEXCOM_AUTH_URL)

    print(f"\n1. Opening browser (or copy URL manually):\n   {authorization_url}\n")
    try:
        webbrowser.open(authorization_url)
    except Exception:
        pass

    redirect_response = input("2. Paste the full redirect URL here: ").strip()

    code = _extract_code(redirect_response)
    token = _exchange_code_dexcom(code)

    Path(cfg.TOKEN_DIR).mkdir(parents=True, exist_ok=True)
    with open(cfg.DEXCOM_TOKEN_FILE, "w") as f:
        json.dump(token, f)
    print(f"Token saved to {cfg.DEXCOM_TOKEN_FILE}")
    log.info("Dexcom token saved to %s", cfg.DEXCOM_TOKEN_FILE)

    # Quick connection test
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    resp = requests.get(f"{cfg.DEXCOM_BASE_URL}users/self/devices", headers=headers)
    if resp.ok:
        devices = resp.json().get("devices", [])
        print(f"Connected! {len(devices)} device(s) found.")
    else:
        print(f"Warning: connection test returned {resp.status_code} {resp.text}")

    return token


# ── Entry point ───────────────────────────────────────────────────────────────

# Keep old name for backwards compatibility
run_oauth_flow = run_oura_oauth_flow

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider = sys.argv[1].lower() if len(sys.argv) > 1 else "oura"
    if provider == "dexcom":
        run_dexcom_oauth_flow()
    elif provider == "oura":
        run_oura_oauth_flow()
    else:
        print(f"Unknown provider '{provider}'. Use 'oura' or 'dexcom'.")
        sys.exit(1)

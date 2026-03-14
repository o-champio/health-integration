"""Interactive OAuth2 flow for the Oura API.

Usage from project root:
    python -m auth.oauth
"""
from __future__ import annotations

import json
import logging
import os
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


def _exchange_code(code: str) -> dict:
    """POST the auth code to Oura's token endpoint and return the token dict."""
    resp = requests.post(
        cfg.TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.OURA_REDIRECT_URL,
            "client_id": cfg.OURA_CLIENT_ID,
            "client_secret": cfg.OURA_CLIENT_SECRET,
        },
    )
    if resp.status_code != 200:
        log.error("Token exchange failed: %s", resp.text)
    resp.raise_for_status()
    return resp.json()


def run_oauth_flow() -> dict:
    """Run the interactive browser-based OAuth2 code-grant flow.

    Returns the token dict and saves it to disk.
    """
    from requests_oauthlib import OAuth2Session

    oura = OAuth2Session(
        cfg.OURA_CLIENT_ID,
        scope=cfg.SCOPES,
        redirect_uri=cfg.OURA_REDIRECT_URL,
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
    token = _exchange_code(code)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_oauth_flow()

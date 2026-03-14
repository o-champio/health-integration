"""Configuration package.

Re-exports all settings so callers can do:
    from config import OURA_CLIENT_ID, TOKEN_FILE, ...
"""
from config.settings import *  # noqa: F401,F403

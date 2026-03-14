import secrets as _s

# Credentials (loaded from secrets.py)
OURA_CLIENT_ID = _s.OURA_CLIENT_ID
OURA_CLIENT_SECRET = _s.OURA_CLIENT_SECRET

# Oura API endpoints
OURA_REDIRECT_URL = "http://localhost:8080"
AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL = "https://api.ouraring.com/oauth/token"
BASE_URL = "https://api.ouraring.com/v2/usercollection/"

# Scopes required for HRV/glucose analysis
SCOPES = [
    "daily",      # Readiness and sleep summaries
    "heartrate",  # HRV and stress data
    "personal",   # Profile data
    "workout",    # Exercise impact on glycemia
    "tag"         # Insulin/carb annotations
]

# Token file (auto-saved and refreshed)
TOKEN_FILE = "oura_token.json"

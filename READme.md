# health-integration

Integration of health data from Oura with glucose data from Libre/Dexcom. Statistical analysis.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install requests-oauthlib
   ```

2. Create a `secrets.py` file in the project root (see `secrets.example.py`):
   ```python
   OURA_CLIENT_ID = "your-client-id"
   OURA_CLIENT_SECRET = "your-client-secret"
   ```

3. Run the auth flow:
   ```bash
   python auth.py
   ```

## Project structure

| File | Description |
|------|-------------|
| `auth.py` | OAuth2 handshake and session management for Oura API |
| `config.py` | Public configuration — API endpoints, scopes, token path |
| `secrets.py` | Private credentials — **gitignored, never committed** |

## Data sources

- **Oura Ring** — HRV, readiness, sleep, heart rate
- **Libre / Dexcom** *(planned)* — continuous glucose monitoring

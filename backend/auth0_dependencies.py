import os
import requests
from cachetools import TTLCache
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from requests import RequestException

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
AUTH0_ISSUER = os.getenv("AUTH0_ISSUER")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")  # optional
if not AUTH0_DOMAIN or not AUTH0_AUDIENCE or not AUTH0_ISSUER:
    raise RuntimeError("AUTH0_DOMAIN, AUTH0_AUDIENCE, and AUTH0_ISSUER must be set")

# Normalize issuer to include trailing slash

if not AUTH0_ISSUER.endswith("/"):
    AUTH0_ISSUER += "/"

JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# Cache JWKS for 10 minutes (tune as needed)

_jwks_cache = TTLCache(maxsize=2, ttl=600)
bearer = HTTPBearer(auto_error=False)
def _get_jwks(force_refresh: bool = False) -> dict:
    if not force_refresh:
        jwks = _jwks_cache.get("jwks")
        if jwks:
            return jwks
    try:
        resp = requests.get(JWKS_URL, timeout=5)
        resp.raise_for_status()
        jwks = resp.json()
    except RequestException:
        raise HTTPException(
            status_code=503,
            detail={"error": "jwks_unavailable", "message": "Auth key service unavailable."},
        )
    _jwks_cache["jwks"] = jwks
    return jwks

def _get_rsa_key(token: str) -> dict:
    header = jwt.get_unverified_header(token)

    if header.get("alg") != "RS256":
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Invalid token algorithm."},
        )
    kid = header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Missing kid header."},
        )

    # Try cached keys
    
    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    # Key rotation: force refresh once
    
    jwks = _get_jwks(force_refresh=True)
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    raise HTTPException(
        status_code=401,
        detail={"error": "invalid_token", "message": "Unknown signing key (kid)."},
    )

def get_current_user_optional(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict | None:
    """
    Returns:
      - None if no Authorization header is present (anonymous)
      - {"user_id": <sub>, "claims": <jwt payload>, "scopes": set(...)} if valid token
    """
    if not creds or not creds.credentials:
        return None
    token = creds.credentials
    rsa_key = _get_rsa_key(token)
    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Token is invalid or expired."},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Token missing subject (sub)."},
        )

    # Optional: SPA authorized party check
    
    if AUTH0_CLIENT_ID:
        azp = payload.get("azp")
        if azp and azp != AUTH0_CLIENT_ID:
            raise HTTPException(
                status_code=401,
                detail={"error": "invalid_token", "message": "Invalid authorized party."},
            )
    scope_str = payload.get("scope", "")
    scopes = set(scope_str.split()) if isinstance(scope_str, str) else set()
    return {"user_id": user_id, "claims": payload, "scopes": scopes}
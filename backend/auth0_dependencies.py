from __future__ import annotations

"""
Shared Auth0 authentication dependencies for FastAPI.

Responsibilities:
- validate Bearer JWTs issued by Auth0
- fetch and cache JWKS for signature verification
- expose FastAPI dependencies for optional and required authentication
- return normalized authenticated-user context only
- enforce timeout / JWKS refresh safeguards
- remain free of feature-specific business rules

Non-responsibilities:
- route authorization policy beyond scope checks
- request/response envelope construction
- user persistence or profile lookup
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Set

import requests
from cachetools import TTLCache
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from requests import RequestException


DEFAULT_JWKS_CACHE_TTL_SECONDS = int(os.getenv("AUTH0_JWKS_CACHE_TTL_SECONDS", "600"))
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("AUTH0_TIMEOUT_SECONDS", "5"))


@dataclass(frozen=True)
class Auth0Config:
    """
    Low-level Auth0 verification configuration.

    Notes:
    - domain defaults to AUTH0_DOMAIN
    - audience defaults to AUTH0_AUDIENCE
    - issuer defaults to AUTH0_ISSUER
    - client_id is optional and enables azp validation when present
    """

    domain: Optional[str] = os.getenv("AUTH0_DOMAIN")
    audience: Optional[str] = os.getenv("AUTH0_AUDIENCE")
    issuer: Optional[str] = os.getenv("AUTH0_ISSUER")
    client_id: Optional[str] = os.getenv("AUTH0_CLIENT_ID")
    jwks_cache_ttl_seconds: int = DEFAULT_JWKS_CACHE_TTL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class AuthenticatedUser:
    """
    Normalized authenticated-user context returned by dependencies.
    """

    user_id: str
    claims: dict[str, Any]
    scopes: Set[str]


class Auth0DependencyProvider:
    """
    Shared low-level Auth0 dependency provider.

    Public contract:
    - get_current_user_optional(...) -> AuthenticatedUser | None
    - get_current_user(...) -> AuthenticatedUser
    - require_scopes(*required_scopes) -> dependency callable
    """

    def __init__(self, config: Optional[Auth0Config] = None) -> None:
        self.config = config or Auth0Config()

        self._domain = self._normalize_domain(self.config.domain)
        self._audience = self._normalize_required_setting(
            self.config.audience,
            field_name="AUTH0_AUDIENCE",
        )
        self._issuer = self._normalize_issuer(self.config.issuer)
        self._client_id = self._normalize_optional_setting(self.config.client_id)
        self._request_timeout_seconds = self._normalize_timeout(
            self.config.request_timeout_seconds
        )
        self._jwks_cache = TTLCache(
            maxsize=2,
            ttl=self._normalize_cache_ttl(self.config.jwks_cache_ttl_seconds),
        )
        self._bearer = HTTPBearer(auto_error=False)

    @property
    def jwks_url(self) -> str:
        return f"https://{self._domain}/.well-known/jwks.json"

    def get_current_user_optional(
        self,
        creds: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    ) -> AuthenticatedUser | None:
        """
        Returns:
        - None when Authorization header is absent
        - AuthenticatedUser when a valid Bearer token is present
        """
        if not creds or not creds.credentials:
            return None

        token = self._normalize_token(creds.credentials)
        rsa_key = self._get_rsa_key(token)

        try:
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
            )
        except JWTError as exc:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Token is invalid or expired.",
                },
            ) from exc

        user_id = self._extract_subject(payload)
        self._validate_authorized_party(payload)
        scopes = self._extract_scopes(payload)

        return AuthenticatedUser(
            user_id=user_id,
            claims=payload,
            scopes=scopes,
        )

    def get_current_user(
        self,
        creds: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    ) -> AuthenticatedUser:
        """
        Returns:
        - AuthenticatedUser when a valid Bearer token is present

        Raises:
        - 401 when Authorization header is missing or invalid
        """
        user = self.get_current_user_optional(creds)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "authorization_required",
                    "message": "Authorization credentials are required.",
                },
            )
        return user

    def require_scopes(self, *required_scopes: str):
        """
        Build a FastAPI dependency that enforces one or more scopes.

        Usage:
            @router.get("/private")
            def private_route(
                current_user: AuthenticatedUser = Depends(auth0.require_scopes("read:items"))
            ):
                ...
        """
        normalized_required_scopes = {
            self._normalize_scope(scope) for scope in required_scopes if str(scope).strip()
        }

        def dependency(
            current_user: AuthenticatedUser = Depends(self.get_current_user),
        ) -> AuthenticatedUser:
            missing_scopes = normalized_required_scopes - current_user.scopes
            if missing_scopes:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "insufficient_scope",
                        "message": "Token does not include the required scopes.",
                        "required_scopes": sorted(normalized_required_scopes),
                        "granted_scopes": sorted(current_user.scopes),
                        "missing_scopes": sorted(missing_scopes),
                    },
                )
            return current_user

        return dependency

    def _get_jwks(self, *, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh:
            cached = self._jwks_cache.get("jwks")
            if cached:
                return cached

        try:
            response = requests.get(
                self.jwks_url,
                timeout=self._request_timeout_seconds,
            )
            response.raise_for_status()
            jwks = response.json()
        except RequestException as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "jwks_unavailable",
                    "message": "Auth key service unavailable.",
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "jwks_invalid",
                    "message": "Auth key service returned invalid JWKS content.",
                },
            ) from exc

        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "jwks_invalid",
                    "message": "Auth key service returned malformed JWKS content.",
                },
            )

        self._jwks_cache["jwks"] = jwks
        return jwks

    def _get_rsa_key(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except JWTError as exc:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Token header is unreadable.",
                },
            ) from exc

        alg = header.get("alg")
        if alg != "RS256":
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Invalid token algorithm.",
                },
            )

        kid = header.get("kid")
        if not kid:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Missing kid header.",
                },
            )

        jwks = self._get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key

        jwks = self._get_jwks(force_refresh=True)
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key

        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_token",
                "message": "Unknown signing key (kid).",
            },
        )

    def _extract_subject(self, payload: dict[str, Any]) -> str:
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Token missing subject (sub).",
                },
            )
        return subject.strip()

    def _validate_authorized_party(self, payload: dict[str, Any]) -> None:
        if not self._client_id:
            return

        azp = payload.get("azp")
        if azp is None:
            return

        if not isinstance(azp, str) or azp.strip() != self._client_id:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Invalid authorized party.",
                },
            )

    @staticmethod
    def _extract_scopes(payload: dict[str, Any]) -> Set[str]:
        scope_value = payload.get("scope", "")
        if not isinstance(scope_value, str):
            return set()
        return {scope.strip() for scope in scope_value.split() if scope.strip()}

    @staticmethod
    def _normalize_token(token: str) -> str:
        if not isinstance(token, str):
            raise TypeError("token must be a string.")
        normalized = token.strip()
        if not normalized:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Bearer token is empty.",
                },
            )
        return normalized

    @staticmethod
    def _normalize_domain(domain: Optional[str]) -> str:
        normalized = Auth0DependencyProvider._normalize_required_setting(
            domain,
            field_name="AUTH0_DOMAIN",
        )
        normalized = normalized.replace("https://", "").replace("http://", "").strip().strip("/")
        if not normalized:
            raise RuntimeError(
                "AUTH0_DOMAIN is not configured. Set it in the environment before using Auth0DependencyProvider."
            )
        return normalized

    @staticmethod
    def _normalize_issuer(issuer: Optional[str]) -> str:
        normalized = Auth0DependencyProvider._normalize_required_setting(
            issuer,
            field_name="AUTH0_ISSUER",
        )
        return normalized if normalized.endswith("/") else f"{normalized}/"

    @staticmethod
    def _normalize_required_setting(value: Optional[str], *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f"{field_name} is not configured. Set it in the environment before using Auth0DependencyProvider."
            )
        return value.strip()

    @staticmethod
    def _normalize_optional_setting(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_timeout(value: float) -> float:
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("request_timeout_seconds must be a numeric value.") from exc

        if normalized <= 0:
            raise ValueError("request_timeout_seconds must be > 0.")
        return normalized

    @staticmethod
    def _normalize_cache_ttl(value: int) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("jwks_cache_ttl_seconds must be an int-like value.") from exc

        if normalized < 1:
            raise ValueError("jwks_cache_ttl_seconds must be >= 1.")
        return normalized

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        if not isinstance(scope, str):
            raise TypeError("scope must be a string.")
        normalized = scope.strip()
        if not normalized:
            raise ValueError("scope must not be empty.")
        return normalized

bearer = HTTPBearer(auto_error=False)

_auth0_provider: Auth0DependencyProvider | None = None


def get_auth0_provider() -> Auth0DependencyProvider:
    global _auth0_provider
    if _auth0_provider is None:
        _auth0_provider = Auth0DependencyProvider()
    return _auth0_provider


def get_current_user_optional(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> AuthenticatedUser | None:
    if not creds or not creds.credentials:
        return None
    return get_auth0_provider().get_current_user_optional(creds)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> AuthenticatedUser:
    if not creds or not creds.credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "authorization_required",
                "message": "Authorization credentials are required.",
            },
        )
    return get_auth0_provider().get_current_user(creds)


def require_scopes(*required_scopes: str):
    def dependency(
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        provider = get_auth0_provider()
        normalized_required_scopes = {
            provider._normalize_scope(scope)
            for scope in required_scopes
            if str(scope).strip()
        }
        missing_scopes = normalized_required_scopes - current_user.scopes
        if missing_scopes:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "insufficient_scope",
                    "message": "Token does not include the required scopes.",
                    "required_scopes": sorted(normalized_required_scopes),
                    "granted_scopes": sorted(current_user.scopes),
                    "missing_scopes": sorted(missing_scopes),
                },
            )
        return current_user

    return dependency


__all__ = [
    "Auth0Config",
    "AuthenticatedUser",
    "Auth0DependencyProvider",
    "bearer",
    "get_auth0_provider",
    "get_current_user_optional",
    "get_current_user",
    "require_scopes",
]
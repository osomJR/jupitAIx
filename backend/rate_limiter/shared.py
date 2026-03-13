from __future__ import annotations

"""
Shared rate-limiting layer for anonymous and authenticated-free users.

Design goals:
- deterministic, Redis-backed, and distributed
- stricter than the legacy limiters
- consistent with the surrounding codebase style
- reusable across tier-specific wrappers to avoid duplication
- hardened against trivial evasion, while remaining honest about limits:
  anonymous traffic can be made more expensive to evade, but not impossible to evade
  with VPNs / proxies / device churn alone at the application layer

Important note:
- authenticated-free protection should key primarily on the verified user subject
- anonymous protection should use a stable multi-signal fingerprint plus a network bucket
- for truly hostile anonymous abuse, pair this with bot mitigation / CAPTCHA / proof-of-work
"""

from dataclasses import dataclass
import hashlib
import hmac
import ipaddress
import os
import time
from typing import Iterable, Optional
from uuid import uuid4

import redis
from fastapi import HTTPException, Request

from src.schema import (
    ANONYMOUS_USER_MAX_ACTIONS_PER_DAY,
    AUTHENTICATED_USER_MAX_ACTIONS_PER_DAY,
    FeatureType,
)


SECONDS_IN_DAY = 86_400
DEFAULT_BURST_LIMIT = int(os.getenv("RATE_LIMIT_BURST_LIMIT", "2"))
DEFAULT_BURST_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_BURST_WINDOW_SECONDS", "10"))
DEFAULT_NETWORK_BURST_LIMIT = int(os.getenv("RATE_LIMIT_NETWORK_BURST_LIMIT", "8"))
DEFAULT_NETWORK_BURST_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_NETWORK_BURST_WINDOW_SECONDS", "10"))
DEFAULT_REDIS_URL_ENV = "REDIS_URL"
DEFAULT_SECRET_ENV = "RATE_LIMIT_HMAC_SECRET"
DEFAULT_DEVICE_COOKIE_NAME = os.getenv("RATE_LIMIT_DEVICE_COOKIE_NAME", "rl_device")
DEFAULT_DEVICE_HEADER_NAME = os.getenv("RATE_LIMIT_DEVICE_HEADER_NAME", "x-device-id")
DEFAULT_SESSION_HEADER_NAME = os.getenv("RATE_LIMIT_SESSION_HEADER_NAME", "x-session-id")
DEFAULT_FAIL_CLOSED = os.getenv("RATE_LIMIT_FAIL_CLOSED", "true").strip().lower() not in {"0", "false", "no"}


LIGHT_FEATURES = frozenset(
    {
        FeatureType.summarize,
        FeatureType.explain,
        FeatureType.translate,
        FeatureType.grammar_correct,
    }
)

HEAVY_FEATURES = frozenset(
    {
        FeatureType.convert,
        FeatureType.generate_questions,
        FeatureType.generate_answers,
        FeatureType.transcribe,
    }
)

ANONYMOUS_ALLOWED_LIGHT_FEATURES = LIGHT_FEATURES
ANONYMOUS_ALLOWED_HEAVY_FEATURES = frozenset({FeatureType.convert})
ANONYMOUS_BLOCKED_FEATURES = frozenset(
    {
        FeatureType.generate_questions,
        FeatureType.generate_answers,
        FeatureType.transcribe,
    }
)

AUTHENTICATED_FREE_ALLOWED_LIGHT_FEATURES = LIGHT_FEATURES
AUTHENTICATED_FREE_ALLOWED_HEAVY_FEATURES = HEAVY_FEATURES
AUTHENTICATED_FREE_BLOCKED_FEATURES = frozenset()


@dataclass(frozen=True)
class RateLimitPolicy:
    tier_name: str
    total_limit: int
    heavy_limit: int
    burst_limit: int = DEFAULT_BURST_LIMIT
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS
    network_total_limit: Optional[int] = None
    network_heavy_limit: Optional[int] = None
    network_burst_limit: Optional[int] = DEFAULT_NETWORK_BURST_LIMIT
    network_burst_window_seconds: int = DEFAULT_NETWORK_BURST_WINDOW_SECONDS
    total_window_seconds: int = SECONDS_IN_DAY
    heavy_window_seconds: int = SECONDS_IN_DAY
    fail_closed: bool = DEFAULT_FAIL_CLOSED

    def __post_init__(self) -> None:
        for field_name in (
            "total_limit",
            "heavy_limit",
            "burst_limit",
            "burst_window_seconds",
            "total_window_seconds",
            "heavy_window_seconds",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an int >= 1.")

        for field_name in ("network_total_limit", "network_heavy_limit", "network_burst_limit"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{field_name} must be None or an int >= 1.")

        if not isinstance(self.tier_name, str) or not self.tier_name.strip():
            raise ValueError("tier_name must be a non-empty string.")


@dataclass(frozen=True)
class LimitOutcome:
    allowed: bool
    current_count: int
    retry_after_seconds: int


class RedisSlidingWindowLimiter:
    """
    Atomic Redis-backed sliding-window limiter.

    Uses one sorted set per logical bucket and enforces limits through Lua so the
    prune/count/add sequence is atomic across concurrent workers.
    """

    _LUA_ENFORCE = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window_seconds = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local member = ARGV[4]

    redis.call('ZREMRANGEBYSCORE', key, 0, now - window_seconds)
    local current_count = redis.call('ZCARD', key)

    if current_count >= limit then
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local retry_after = window_seconds
        if oldest[2] ~= nil then
            retry_after = math.max(1, math.floor(window_seconds - (now - tonumber(oldest[2]))))
        end
        return {0, current_count, retry_after}
    end

    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window_seconds)
    return {1, current_count + 1, 0}
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis_client = redis_client
        self._script = self.redis_client.register_script(self._LUA_ENFORCE)

    def enforce(self, *, key: str, limit: int, window_seconds: int, now: Optional[int] = None) -> LimitOutcome:
        timestamp = int(now or time.time())
        member = f"{timestamp}:{uuid4().hex}"
        raw = self._script(keys=[key], args=[timestamp, window_seconds, limit, member])
        allowed = bool(int(raw[0]))
        current_count = int(raw[1])
        retry_after_seconds = int(raw[2])
        return LimitOutcome(
            allowed=allowed,
            current_count=current_count,
            retry_after_seconds=retry_after_seconds,
        )


class RateLimitShared:
    """
    Shared orchestrator used by anonymous and authenticated-free wrappers.
    """

    def __init__(
        self,
        *,
        redis_client: Optional[redis.Redis] = None,
        secret: Optional[str] = None,
        device_cookie_name: str = DEFAULT_DEVICE_COOKIE_NAME,
        device_header_name: str = DEFAULT_DEVICE_HEADER_NAME,
        session_header_name: str = DEFAULT_SESSION_HEADER_NAME,
    ) -> None:
        self.redis_client = redis_client or get_redis_client()
        self.limiter = RedisSlidingWindowLimiter(self.redis_client)
        self.secret = self._normalize_secret(secret or os.getenv(DEFAULT_SECRET_ENV))
        self.device_cookie_name = self._normalize_header_or_cookie_name(device_cookie_name, field_name="device_cookie_name")
        self.device_header_name = self._normalize_header_or_cookie_name(device_header_name, field_name="device_header_name")
        self.session_header_name = self._normalize_header_or_cookie_name(session_header_name, field_name="session_header_name")

    def enforce_anonymous(
        self,
        *,
        request: Request,
        feature: FeatureType,
        policy: RateLimitPolicy,
        allowed_light_features: Iterable[FeatureType],
        allowed_heavy_features: Iterable[FeatureType],
        blocked_features: Iterable[FeatureType],
        family: str,
    ) -> None:
        self._validate_feature_for_family(
            feature=feature,
            family=family,
            allowed_light_features=allowed_light_features,
            allowed_heavy_features=allowed_heavy_features,
            blocked_features=blocked_features,
            tier_name=policy.tier_name,
        )

        actor_hash = self._anonymous_actor_hash(request)
        network_hash = self._network_hash(request)
        now = int(time.time())

        try:
            self._enforce_policy_dimensions(
                policy=policy,
                now=now,
                actor_key_prefix=f"rl:v2:{policy.tier_name}:anon:{actor_hash}",
                network_key_prefix=(
                    f"rl:v2:{policy.tier_name}:network:{network_hash}"
                    if network_hash is not None else None
                ),
                feature=feature,
            )
        except redis.RedisError as exc:
            self._handle_backend_failure(policy=policy, exc=exc)

    def enforce_authenticated_free(
        self,
        *,
        request: Request,
        user_id: str,
        feature: FeatureType,
        policy: RateLimitPolicy,
        allowed_light_features: Iterable[FeatureType],
        allowed_heavy_features: Iterable[FeatureType],
        blocked_features: Iterable[FeatureType],
        family: str,
    ) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "authorization_required",
                    "message": "Authentication credentials are required.",
                },
            )

        self._validate_feature_for_family(
            feature=feature,
            family=family,
            allowed_light_features=allowed_light_features,
            allowed_heavy_features=allowed_heavy_features,
            blocked_features=blocked_features,
            tier_name=policy.tier_name,
        )

        actor_hash = self._hash_value(f"user:{user_id.strip()}")
        device_hash = self._optional_device_hash(request)
        now = int(time.time())

        try:
            self._enforce_policy_dimensions(
                policy=policy,
                now=now,
                actor_key_prefix=f"rl:v2:{policy.tier_name}:user:{actor_hash}",
                network_key_prefix=(
                    f"rl:v2:{policy.tier_name}:device:{device_hash}"
                    if device_hash is not None else None
                ),
                feature=feature,
            )
        except redis.RedisError as exc:
            self._handle_backend_failure(policy=policy, exc=exc)

    def _enforce_policy_dimensions(
        self,
        *,
        policy: RateLimitPolicy,
        now: int,
        actor_key_prefix: str,
        network_key_prefix: Optional[str],
        feature: FeatureType,
    ) -> None:
        is_heavy = _is_heavy_feature(feature)

        self._enforce_window(
            key=f"{actor_key_prefix}:burst",
            limit=policy.burst_limit,
            window_seconds=policy.burst_window_seconds,
            error="burst_limit_exceeded",
            message="Too many requests in a short time. Slow down.",
            retry_scope="burst",
            now=now,
        )
        self._enforce_window(
            key=f"{actor_key_prefix}:total",
            limit=policy.total_limit,
            window_seconds=policy.total_window_seconds,
            error="rate_limit_exceeded",
            message=(
                f"{policy.tier_name.replace('_', ' ').title()} usage limit reached "
                f"({policy.total_limit}/{_window_label(policy.total_window_seconds)})."
            ),
            retry_scope="daily_total",
            now=now,
        )

        if is_heavy:
            self._enforce_window(
                key=f"{actor_key_prefix}:heavy",
                limit=policy.heavy_limit,
                window_seconds=policy.heavy_window_seconds,
                error="heavy_rate_limit_exceeded",
                message=(
                    f"{policy.tier_name.replace('_', ' ').title()} heavy-action limit reached "
                    f"({policy.heavy_limit}/{_window_label(policy.heavy_window_seconds)})."
                ),
                retry_scope="daily_heavy",
                now=now,
            )

        if network_key_prefix is None:
            return

        if policy.network_burst_limit is not None:
            self._enforce_window(
                key=f"{network_key_prefix}:burst",
                limit=policy.network_burst_limit,
                window_seconds=policy.network_burst_window_seconds,
                error="network_burst_limit_exceeded",
                message="Too many requests are arriving from this client cluster. Slow down.",
                retry_scope="network_burst",
                now=now,
            )

        if policy.network_total_limit is not None:
            self._enforce_window(
                key=f"{network_key_prefix}:total",
                limit=policy.network_total_limit,
                window_seconds=policy.total_window_seconds,
                error="network_rate_limit_exceeded",
                message="This client cluster has reached its usage allowance. Try again later.",
                retry_scope="network_total",
                now=now,
            )

        if is_heavy and policy.network_heavy_limit is not None:
            self._enforce_window(
                key=f"{network_key_prefix}:heavy",
                limit=policy.network_heavy_limit,
                window_seconds=policy.heavy_window_seconds,
                error="network_heavy_rate_limit_exceeded",
                message="This client cluster has reached its heavy-action allowance. Try again later.",
                retry_scope="network_heavy",
                now=now,
            )

    def _enforce_window(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int,
        error: str,
        message: str,
        retry_scope: str,
        now: int,
    ) -> None:
        outcome = self.limiter.enforce(key=key, limit=limit, window_seconds=window_seconds, now=now)
        if outcome.allowed:
            return

        retry_after = max(1, int(outcome.retry_after_seconds or window_seconds))
        raise HTTPException(
            status_code=429,
            detail={
                "error": error,
                "message": message,
                "retry_after_seconds": retry_after,
                "scope": retry_scope,
                "limit": limit,
                "window_seconds": window_seconds,
            },
            headers={"Retry-After": str(retry_after)},
        )

    def _anonymous_actor_hash(self, request: Request) -> str:
        network = self._extract_client_network_token(request)
        device = self._extract_device_token(request)
        headers = request.headers

        components = [
            f"network={network or 'unknown'}",
            f"device={device or 'none'}",
            f"ua={headers.get('user-agent', '').strip()}",
            f"lang={headers.get('accept-language', '').strip()}",
            f"enc={headers.get('accept-encoding', '').strip()}",
            f"sec_ch_ua={headers.get('sec-ch-ua', '').strip()}",
            f"sec_platform={headers.get('sec-ch-ua-platform', '').strip()}",
            f"sec_mobile={headers.get('sec-ch-ua-mobile', '').strip()}",
        ]
        return self._hash_value("|".join(components))

    def _network_hash(self, request: Request) -> Optional[str]:
        network = self._extract_client_network_token(request)
        if network is None:
            return None
        return self._hash_value(f"network:{network}")

    def _optional_device_hash(self, request: Request) -> Optional[str]:
        token = self._extract_device_token(request)
        if token is None:
            return None
        return self._hash_value(f"device:{token}")

    def _extract_device_token(self, request: Request) -> Optional[str]:
        header_value = request.headers.get(self.device_header_name)
        if header_value and header_value.strip():
            return header_value.strip()[:256]

        session_value = request.headers.get(self.session_header_name)
        if session_value and session_value.strip():
            return session_value.strip()[:256]

        cookie_value = request.cookies.get(self.device_cookie_name)
        if cookie_value and cookie_value.strip():
            return cookie_value.strip()[:256]

        return None

    def _extract_client_network_token(self, request: Request) -> Optional[str]:
        candidate = _extract_client_ip(request)
        if not candidate:
            return None

        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            return None

        if isinstance(address, ipaddress.IPv4Address):
            network = ipaddress.ip_network(f"{address}/24", strict=False)
            return str(network.network_address) + "/24"

        network = ipaddress.ip_network(f"{address}/64", strict=False)
        return str(network.network_address) + "/64"

    def _validate_feature_for_family(
        self,
        *,
        feature: FeatureType,
        family: str,
        allowed_light_features: Iterable[FeatureType],
        allowed_heavy_features: Iterable[FeatureType],
        blocked_features: Iterable[FeatureType],
        tier_name: str,
    ) -> None:
        blocked = set(blocked_features)
        if feature in blocked:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "feature_not_available",
                    "message": f"{feature.value} is not available for {tier_name.replace('_', ' ')} users.",
                },
            )

        allowed_light = set(allowed_light_features)
        allowed_heavy = set(allowed_heavy_features)

        normalized_family = family.strip().lower()
        if normalized_family == "light":
            if feature not in allowed_light:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_light_feature",
                        "message": f"{feature.value} is not configured as a light feature for {tier_name}.",
                    },
                )
            return

        if normalized_family == "heavy":
            if feature not in allowed_heavy:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_heavy_feature",
                        "message": f"{feature.value} is not configured as a heavy feature for {tier_name}.",
                    },
                )
            return

        raise ValueError("family must be either 'light' or 'heavy'.")

    def _handle_backend_failure(self, *, policy: RateLimitPolicy, exc: Exception) -> None:
        if policy.fail_closed:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "rate_limit_backend_unavailable",
                    "message": "Rate-limit backend is temporarily unavailable.",
                },
            ) from exc
        return

    def _hash_value(self, value: str) -> str:
        payload = value.encode("utf-8")
        return hmac.new(self.secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    @staticmethod
    def _normalize_secret(secret: Optional[str]) -> str:
        if not isinstance(secret, str) or not secret.strip():
            raise RuntimeError(
                f"{DEFAULT_SECRET_ENV} is not configured. Set it before using the rate limiter."
            )
        return secret.strip()

    @staticmethod
    def _normalize_header_or_cookie_name(value: str, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip().lower()


def get_redis_client() -> redis.Redis:
    redis_url = os.getenv(DEFAULT_REDIS_URL_ENV)
    if not isinstance(redis_url, str) or not redis_url.strip():
        raise RuntimeError(f"{DEFAULT_REDIS_URL_ENV} is not set")

    normalized = redis_url.strip()
    if not normalized.startswith("rediss://"):
        raise RuntimeError(f"{DEFAULT_REDIS_URL_ENV} must use rediss:// (TLS required)")

    client = redis.from_url(normalized, decode_responses=False)
    try:
        client.ping()
    except redis.RedisError as exc:
        raise RuntimeError(f"Redis connection failed: {exc}") from exc
    return client


def _extract_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        first = forwarded.split(",")[0].strip()
        if first:
            return first

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    if request.client and request.client.host:
        return str(request.client.host).strip() or None

    return None


def _is_heavy_feature(feature: FeatureType) -> bool:
    return feature in HEAVY_FEATURES


def _window_label(window_seconds: int) -> str:
    if window_seconds == SECONDS_IN_DAY:
        return "day"
    if window_seconds % 3600 == 0:
        hours = window_seconds // 3600
        return f"{hours}h"
    if window_seconds % 60 == 0:
        minutes = window_seconds // 60
        return f"{minutes}m"
    return f"{window_seconds}s"


ANONYMOUS_POLICY = RateLimitPolicy(
    tier_name="anonymous",
    total_limit=ANONYMOUS_USER_MAX_ACTIONS_PER_DAY,
    heavy_limit=2,
    burst_limit=2,
    burst_window_seconds=10,
    network_total_limit=int(os.getenv("ANONYMOUS_NETWORK_TOTAL_LIMIT", "12")),
    network_heavy_limit=int(os.getenv("ANONYMOUS_NETWORK_HEAVY_LIMIT", "4")),
    network_burst_limit=int(os.getenv("ANONYMOUS_NETWORK_BURST_LIMIT", "8")),
    network_burst_window_seconds=int(os.getenv("ANONYMOUS_NETWORK_BURST_WINDOW_SECONDS", "10")),
)

AUTHENTICATED_FREE_POLICY = RateLimitPolicy(
    tier_name="authenticated_free",
    total_limit=AUTHENTICATED_USER_MAX_ACTIONS_PER_DAY,
    heavy_limit=3,
    burst_limit=2,
    burst_window_seconds=10,
    network_total_limit=None,
    network_heavy_limit=None,
    network_burst_limit=int(os.getenv("AUTHENTICATED_FREE_DEVICE_BURST_LIMIT", "6")),
    network_burst_window_seconds=int(os.getenv("AUTHENTICATED_FREE_DEVICE_BURST_WINDOW_SECONDS", "10")),
)

shared_rate_limiter = RateLimitShared()


__all__ = [
    "ANONYMOUS_ALLOWED_HEAVY_FEATURES",
    "ANONYMOUS_ALLOWED_LIGHT_FEATURES",
    "ANONYMOUS_BLOCKED_FEATURES",
    "ANONYMOUS_POLICY",
    "AUTHENTICATED_FREE_ALLOWED_HEAVY_FEATURES",
    "AUTHENTICATED_FREE_ALLOWED_LIGHT_FEATURES",
    "AUTHENTICATED_FREE_BLOCKED_FEATURES",
    "AUTHENTICATED_FREE_POLICY",
    "HEAVY_FEATURES",
    "LIGHT_FEATURES",
    "LimitOutcome",
    "RateLimitPolicy",
    "RateLimitShared",
    "RedisSlidingWindowLimiter",
    "get_redis_client",
    "shared_rate_limiter",
]

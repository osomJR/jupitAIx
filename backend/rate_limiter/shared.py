from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from src.schema import FeatureType


SECONDS_IN_DAY = 24 * 60 * 60

DEFAULT_BURST_LIMIT = int(os.getenv("RATE_LIMIT_BURST_LIMIT", "2"))
DEFAULT_BURST_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_BURST_WINDOW_SECONDS", "10"))
DEFAULT_NETWORK_BURST_LIMIT = int(os.getenv("RATE_LIMIT_NETWORK_BURST_LIMIT", "10"))
DEFAULT_NETWORK_BURST_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_NETWORK_BURST_WINDOW_SECONDS", "10"))
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
        FeatureType.redact,
        FeatureType.data_mask,
    }
)

ANONYMOUS_ALLOWED_LIGHT_FEATURES = LIGHT_FEATURES
ANONYMOUS_ALLOWED_HEAVY_FEATURES = frozenset({FeatureType.convert})
ANONYMOUS_BLOCKED_FEATURES = frozenset(
    {
        FeatureType.generate_questions,
        FeatureType.generate_answers,
        FeatureType.transcribe,
        FeatureType.redact,
        FeatureType.data_mask,
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
      local retry_after = 1
      if oldest[2] ~= nil then
        retry_after = math.max(1, math.floor((oldest[2] + window_seconds) - now))
      end
      return {0, current_count, retry_after}
    end

    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window_seconds)
    return {1, current_count + 1, 0}
    """

    def __init__(self, redis_client) -> None:
        self.redis = redis_client
        self._enforce_script = self.redis.register_script(self._LUA_ENFORCE)

    def enforce(self, *, key: str, limit: int, window_seconds: int) -> LimitOutcome:
        import time
        now = int(time.time())
        member = f"{now}:{os.urandom(8).hex()}"
        allowed, current_count, retry_after = self._enforce_script(
            keys=[key],
            args=[now, window_seconds, limit, member],
        )
        return LimitOutcome(
            allowed=bool(int(allowed)),
            current_count=int(current_count),
            retry_after_seconds=int(retry_after),
        )


def get_shared_rate_limiter():
    raise NotImplementedError("Project-specific rate limiter wiring remains unchanged.")


ANONYMOUS_POLICY = RateLimitPolicy(
    tier_name="anonymous",
    total_limit=4,
    heavy_limit=2,
)

AUTHENTICATED_FREE_POLICY = RateLimitPolicy(
    tier_name="authenticated_free",
    total_limit=7,
    heavy_limit=3,
)


__all__ = [
    "SECONDS_IN_DAY",
    "DEFAULT_BURST_LIMIT",
    "DEFAULT_BURST_WINDOW_SECONDS",
    "DEFAULT_NETWORK_BURST_LIMIT",
    "DEFAULT_NETWORK_BURST_WINDOW_SECONDS",
    "DEFAULT_DEVICE_HEADER_NAME",
    "DEFAULT_SESSION_HEADER_NAME",
    "DEFAULT_FAIL_CLOSED",
    "LIGHT_FEATURES",
    "HEAVY_FEATURES",
    "ANONYMOUS_ALLOWED_LIGHT_FEATURES",
    "ANONYMOUS_ALLOWED_HEAVY_FEATURES",
    "ANONYMOUS_BLOCKED_FEATURES",
    "AUTHENTICATED_FREE_ALLOWED_LIGHT_FEATURES",
    "AUTHENTICATED_FREE_ALLOWED_HEAVY_FEATURES",
    "AUTHENTICATED_FREE_BLOCKED_FEATURES",
    "RateLimitPolicy",
    "LimitOutcome",
    "RedisSlidingWindowLimiter",
    "get_shared_rate_limiter",
    "ANONYMOUS_POLICY",
    "AUTHENTICATED_FREE_POLICY",
]

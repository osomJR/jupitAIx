from fastapi import HTTPException, Request
import time
import redis
import hashlib
import os
from src.schema import FeatureType
"""
AI_RATE_LIMITING 
Memory-based • Distributed • Burst • Hardened Fingerprint
"""

# CONFIG

ANONYMOUS_DAILY_LIMIT = 3
HEAVY_ANONYMOUS_DAILY_LIMIT = 2
BURST_LIMIT = 2
BURST_WINDOW_SECONDS = 10
SECONDS_IN_DAY = 86400

# REDIS

def get_redis_client():
    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")

    if not redis_url.startswith("rediss://"):
        raise RuntimeError("REDIS_URL must use rediss:// (TLS required for Upstash TCP)")

    return redis.from_url(redis_url)

redis_client = get_redis_client()

# Fail fast if credentials are wrong

try:
    redis_client.ping()
except redis.RedisError as e:
    raise RuntimeError(f"Redis connection failed: {e}")

# UTILITIES

def _is_heavy_feature(feature: FeatureType) -> bool:
    return feature in {
        FeatureType.generate_questions,
        FeatureType.generate_answers,
    }

def _fingerprint(request: Request) -> str:
    """
    Stronger SaaS-grade fingerprint:
    Combines:
    - IP (proxy-aware)
    - User-Agent
    - Accept-Language
    - Accept-Encoding
    - Sec-CH-UA
    """

    # Proxy-safe IP resolution
    
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host or ""

    headers = request.headers
    components = [
        ip,
        headers.get("user-agent", ""),
        headers.get("accept-language", ""),
        headers.get("accept-encoding", ""),
        headers.get("sec-ch-ua", ""),
    ]

    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()

# MAIN RATE LIMIT FUNCTION

def rate_limit_ai(request: Request, feature: FeatureType) -> None:
    """
    SaaS MVP Rate Limiter
    - Distributed (Upstash)
    - Memory-based
    - Daily sliding window
    - Burst protection
    """

    now = int(time.time())
    limit = HEAVY_ANONYMOUS_DAILY_LIMIT if _is_heavy_feature(feature) else ANONYMOUS_DAILY_LIMIT

    fingerprint = _fingerprint(request)
    daily_key = f"rl:daily:{fingerprint}"
    burst_key = f"rl:burst:{fingerprint}"

    try:

        # DAILY LIMIT

        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(daily_key, 0, now - SECONDS_IN_DAY)
        pipe.zcard(daily_key)
        results = pipe.execute()

        current_daily = results[1]

        if current_daily >= limit:
            oldest = redis_client.zrange(daily_key, 0, 0, withscores=True)
            retry_after = int(SECONDS_IN_DAY - (now - oldest[0][1])) if oldest else SECONDS_IN_DAY

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Anonymous free usage limit reached ({limit}/day). Upgrade to continue.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Group writes in pipeline
        
        pipe = redis_client.pipeline()
        pipe.zadd(daily_key, {str(now): now})
        pipe.expire(daily_key, SECONDS_IN_DAY)
        pipe.execute()

        # BURST LIMIT

        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(burst_key, 0, now - BURST_WINDOW_SECONDS)
        pipe.zcard(burst_key)
        results = pipe.execute()

        current_burst = results[1]

        if current_burst >= BURST_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "burst_limit_exceeded",
                    "message": "Too many requests in a short time. Slow down.",
                    "retry_after_seconds": BURST_WINDOW_SECONDS,
                },
                headers={"Retry-After": str(BURST_WINDOW_SECONDS)},
            )
  
        # Group writes in pipeline
        
        pipe = redis_client.pipeline()
        pipe.zadd(burst_key, {str(now): now})
        pipe.expire(burst_key, BURST_WINDOW_SECONDS)
        pipe.execute()

    except redis.RedisError:
        
        # Fail-open for MVP
        
        return
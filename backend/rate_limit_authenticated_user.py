from fastapi import HTTPException, Request
import time
import redis
import hashlib
import os

"""
AI_RATE_LIMITING (Authenticated)
Memory-based • Distributed • Burst • User-scoped
"""

# CONFIG

AUTHENTICATED_DAILY_LIMIT_TOTAL = 5
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

def _user_key(user_id: str) -> str:
    """
    Stable, privacy-preserving user key.
    You can also just use user_id directly, but hashing reduces accidental leakage in logs/exports.
    """
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

def rate_limit_authenticated_user(request: Request, user_id: str) -> None:
    """
    Authenticated tier limiter:
    - 5 total requests per day (heavy+normal combined)
    - Burst protection (2 per 10s)
    - Sliding windows via Redis sorted sets
    """
    if not user_id:
        # This should never happen if your auth dependency is correct,
        # but keep it safe.
        raise HTTPException(status_code=401, detail="Unauthorized")
    now = int(time.time())
    user_hash = _user_key(user_id)
    daily_key = f"rl:auth:daily:{user_hash}"
    burst_key = f"rl:auth:burst:{user_hash}"
    try:
        
        # DAILY LIMIT 
        
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(daily_key, 0, now - SECONDS_IN_DAY)
        pipe.zcard(daily_key)
        results = pipe.execute()
        current_daily = results[1]
        if current_daily >= AUTHENTICATED_DAILY_LIMIT_TOTAL:
            oldest = redis_client.zrange(daily_key, 0, 0, withscores=True)
            retry_after = int(SECONDS_IN_DAY - (now - oldest[0][1])) if oldest else SECONDS_IN_DAY
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "upgrade_required",
                    "message": (
                        f"Free account limit reached ({AUTHENTICATED_DAILY_LIMIT_TOTAL}/day). "
                        "Upgrade to continue."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request in daily window
        # Use a unique member to avoid collisions within the same second.
       
        member = f"{now}:{os.urandom(6).hex()}"
        pipe = redis_client.pipeline()
        pipe.zadd(daily_key, {member: now})
        pipe.expire(daily_key, SECONDS_IN_DAY)
        pipe.execute()

        # BURST LIMIT (2/10 seconds)
    
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

        # Record this request in burst window
        
        member_burst = f"{now}:{os.urandom(6).hex()}"
        pipe = redis_client.pipeline()
        pipe.zadd(burst_key, {member_burst: now})
        pipe.expire(burst_key, BURST_WINDOW_SECONDS)
        pipe.execute()
    except redis.RedisError:
        # Fail-open for MVP
        return
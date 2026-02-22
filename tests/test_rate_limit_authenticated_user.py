import time
import pytest
import redis
import fakeredis
from fastapi import HTTPException
from starlette.requests import Request
from starlette.datastructures import Headers
from starlette.types import Scope

# Change this import to match your project structure.
# Example: from app.rate_limit_authenticated_user import rate_limit_authenticated_user, _user_key

import backend.rate_limit_authenticated_user as rl

def make_request(path: str = "/test") -> Request:
    """
    Create a minimal Starlette/FastAPI Request object.
    The limiter doesn't use request currently, but we pass a real one anyway.
    """
    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": Headers({}).raw,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)

@pytest.fixture()
def fake_redis(monkeypatch):
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)

    monkeypatch.setattr(rl, "redis_client", client)
    return client

@pytest.fixture()
def frozen_time(monkeypatch):
    """
    Freeze time.time() used by the limiter.
    We'll set it per-test by assigning frozen_time["now"].
    """
    state = {"now": 1_700_000_000}  # arbitrary stable epoch
    def _fake_time():
        return state["now"]
    monkeypatch.setattr(rl.time, "time", _fake_time)
    return state

def test_missing_user_id_raises_401(fake_redis):
    req = make_request()
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_authenticated_user(req, user_id="")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Unauthorized"

def test_allows_under_limits(fake_redis, frozen_time):
    req = make_request()
    user_id = "user-123"

    # 1st request should pass
   
    rl.rate_limit_authenticated_user(req, user_id=user_id)

    # Verify keys exist (daily + burst)
    
    user_hash = rl._user_key(user_id)
    daily_key = f"rl:auth:daily:{user_hash}"
    burst_key = f"rl:auth:burst:{user_hash}"
    assert fake_redis.zcard(daily_key) == 1
    assert fake_redis.zcard(burst_key) == 1

def test_daily_limit_blocks_on_6th_request(fake_redis, frozen_time):
    # Use a small limit for test speed & clarity
    rl.AUTHENTICATED_DAILY_LIMIT_TOTAL = 5
    rl.SECONDS_IN_DAY = 86400

    # Keep burst as-is, but avoid triggering it by spacing requests out
    rl.BURST_LIMIT = 2
    rl.BURST_WINDOW_SECONDS = 10

    req = make_request()
    user_id = "user-daily"

    # Make 5 requests, but advance time beyond the burst window each time
    start = 1_700_000_000
    for i in range(rl.AUTHENTICATED_DAILY_LIMIT_TOTAL):
        frozen_time["now"] = start + i * (rl.BURST_WINDOW_SECONDS + 1)
        rl.rate_limit_authenticated_user(req, user_id=user_id)

    # 6th request should fail daily limit (still within same day)
    frozen_time["now"] = start + rl.AUTHENTICATED_DAILY_LIMIT_TOTAL * (rl.BURST_WINDOW_SECONDS + 1)
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_authenticated_user(req, user_id=user_id)

    err = exc.value
    assert err.status_code == 429
    assert err.detail["error"] == "upgrade_required"
    assert "Free account limit reached" in err.detail["message"]
    assert "retry_after_seconds" in err.detail
    assert err.headers.get("Retry-After") == str(err.detail["retry_after_seconds"])

def test_daily_retry_after_calculation_is_based_on_oldest(fake_redis, frozen_time):
    rl.AUTHENTICATED_DAILY_LIMIT_TOTAL = 5
    rl.SECONDS_IN_DAY = 86400
    req = make_request()
    user_id = "user-retry-after"
    user_hash = rl._user_key(user_id)
    daily_key = f"rl:auth:daily:{user_hash}"

    # Set "now" and pre-populate exactly 5 entries with known timestamps
    
    base = 1_700_000_000
    frozen_time["now"] = base

    # Oldest entry is base - 100 seconds
    
    oldest_ts = base - 100
    for ts in [oldest_ts, base - 50, base - 10, base - 5, base - 1]:
        fake_redis.zadd(daily_key, {f"{ts}:seed": ts})

    # Next call should be blocked and retry_after should be SECONDS_IN_DAY - (now - oldest_ts)
    
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_authenticated_user(req, user_id=user_id)
    err = exc.value
    assert err.status_code == 429
    expected_retry = int(rl.SECONDS_IN_DAY - (base - oldest_ts))
    assert err.detail["retry_after_seconds"] == expected_retry
    assert err.headers.get("Retry-After") == str(expected_retry)

def test_burst_limit_blocks_on_3rd_request_within_window(fake_redis, frozen_time):
    rl.BURST_LIMIT = 2
    rl.BURST_WINDOW_SECONDS = 10
    req = make_request()
    user_id = "user-burst"
    frozen_time["now"] = 1_700_000_000

    # First two pass
    
    rl.rate_limit_authenticated_user(req, user_id=user_id)
    rl.rate_limit_authenticated_user(req, user_id=user_id)

    # Third within same 10s window should block
    
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_authenticated_user(req, user_id=user_id)
    err = exc.value
    assert err.status_code == 429
    assert err.detail["error"] == "burst_limit_exceeded"
    assert err.detail["retry_after_seconds"] == rl.BURST_WINDOW_SECONDS
    assert err.headers.get("Retry-After") == str(rl.BURST_WINDOW_SECONDS)

def test_burst_window_slides_and_allows_after_window(fake_redis, frozen_time):
    rl.BURST_LIMIT = 2
    rl.BURST_WINDOW_SECONDS = 10
    req = make_request()
    user_id = "user-burst-slide"
    t0 = 1_700_000_000
    frozen_time["now"] = t0
    rl.rate_limit_authenticated_user(req, user_id=user_id)
    rl.rate_limit_authenticated_user(req, user_id=user_id)

    # Move past the burst window, should allow again
    
    frozen_time["now"] = t0 + rl.BURST_WINDOW_SECONDS + 1
    rl.rate_limit_authenticated_user(req, user_id=user_id)

def test_daily_window_slides_and_allows_after_day(fake_redis, frozen_time):
    rl.AUTHENTICATED_DAILY_LIMIT_TOTAL = 5
    rl.SECONDS_IN_DAY = 86400

    rl.BURST_LIMIT = 2
    rl.BURST_WINDOW_SECONDS = 10

    req = make_request()
    user_id = "user-daily-slide"
    user_hash = rl._user_key(user_id)
    daily_key = f"rl:auth:daily:{user_hash}"

    t0 = 1_700_000_000

    # Hit daily limit with requests spaced out beyond burst window
    for i in range(rl.AUTHENTICATED_DAILY_LIMIT_TOTAL):
        frozen_time["now"] = t0 + i * (rl.BURST_WINDOW_SECONDS + 1)
        rl.rate_limit_authenticated_user(req, user_id=user_id)

    # Should be blocked by daily limit now
    frozen_time["now"] = t0 + rl.AUTHENTICATED_DAILY_LIMIT_TOTAL * (rl.BURST_WINDOW_SECONDS + 1)
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_authenticated_user(req, user_id=user_id)
    assert exc.value.detail["error"] == "upgrade_required"

    # Move far enough that ALL previous entries are older than the 24h window
    frozen_time["now"] = t0 + rl.SECONDS_IN_DAY + 60

    # Now it should pass (and old items are trimmed)
    rl.rate_limit_authenticated_user(req, user_id=user_id)

    assert fake_redis.zcard(daily_key) == 1

def test_fail_open_on_redis_error(monkeypatch, frozen_time):
    """
    Your code does: except redis.RedisError: return
    So we ensure it does NOT raise, even if Redis fails mid-flight.
    """
    req = make_request()
    class BrokenRedis:
        def pipeline(self):
            raise redis.RedisError("boom")
    monkeypatch.setattr(rl, "redis_client", BrokenRedis())

    # Should not raise any exception (fail-open)
   
    rl.rate_limit_authenticated_user(req, user_id="user-any")
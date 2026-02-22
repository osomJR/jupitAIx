import pytest
from fastapi import HTTPException
from types import SimpleNamespace
import backend.rate_limit_anonymous_user as rl

# Helpers

# Fixed FakePipeline 

class FakePipeline:
    def __init__(self, redis_store):
        self.redis_store = redis_store
        self.commands = []

    def zremrangebyscore(self, key, start, end):
        self.commands.append(("zremrangebyscore", key, start, end))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def execute(self):
        results = []
        for cmd in self.commands:
            key = cmd[1] if len(cmd) > 1 else None
            if cmd[0] == "zremrangebyscore":
                start, end = cmd[2], cmd[3]
                if key in self.redis_store:
                    self.redis_store[key] = [v for v in self.redis_store[key] if not (start <= v <= end)]
                results.append(0)  # zremrangebyscore returns number removed (simulate 0)
            elif cmd[0] == "zcard":
                results.append(len(self.redis_store.get(key, [])))
            elif cmd[0] == "zadd":
                score = list(cmd[2].values())[0]
                self.redis_store.setdefault(key, []).append(score)
                results.append(1)  # zadd returns number added
            elif cmd[0] == "expire":
                results.append(True)
        self.commands = []
        return results

# Fixed FakeRedis

class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return FakePipeline(self.store)

    def zrange(self, key, start, end, withscores=False):
        values = self.store.get(key, [])
        if not values:
            return []
        if withscores:
            return [(str(values[0]), values[0])]
        return [str(values[0])]
    def ping(self):
        return True
class FakeRequest:
    def __init__(self):
        self.client = SimpleNamespace(host="127.0.0.1")
        self.headers = {}

# Fixtures

@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rl, "redis_client", fake)
    return fake

# Tests

def test_registered_user_bypass(mock_redis):
    request = FakeRequest()
    rl.rate_limit_ai(request, feature=None, user_id="123")
    
    # Should not raise

def test_anonymous_under_daily_limit(mock_redis, monkeypatch):
    request = FakeRequest()

    # Advance beyond burst window (assume 10s window)
    
    times = [1000, 1015, 1030]
    monkeypatch.setattr(
        "backend.rate_limit_anonymous_user.time.time",
        lambda: times.pop(0)
    )
    rl.rate_limit_ai(request, feature=None)
    rl.rate_limit_ai(request, feature=None)
    rl.rate_limit_ai(request, feature=None)  # Still under daily limit

def test_anonymous_exceeds_daily_limit(mock_redis, monkeypatch):
    request = FakeRequest()

    # Each request outside burst window
    
    times = [2000, 2015, 2030, 2045]
    monkeypatch.setattr(
        "backend.rate_limit_anonymous_user.time.time",
        lambda: times.pop(0)
    )
    for _ in range(3):
        rl.rate_limit_ai(request, feature=None)
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_ai(request, feature=None)
    assert exc.value.status_code == 429
    assert exc.value.detail["error"] == "rate_limit_exceeded"

def test_burst_limit(mock_redis):
    request = FakeRequest()
    rl.rate_limit_ai(request, feature=None)
    rl.rate_limit_ai(request, feature=None)
    with pytest.raises(HTTPException) as exc:
        rl.rate_limit_ai(request, feature=None)
    assert exc.value.detail["error"] == "burst_limit_exceeded"

def test_fail_open_on_redis_error(monkeypatch):
    request = FakeRequest()
    def broken_pipeline():
        raise rl.redis.RedisError("fail")
    monkeypatch.setattr(rl.redis_client, "pipeline", broken_pipeline)

    # Should not raise because fail-open
    
    rl.rate_limit_ai(request, feature=None)
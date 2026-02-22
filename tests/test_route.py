import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.schema import FeatureType
from backend.route import router

# Test App Setup

app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Patch rate_limit_ai for all route tests

@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch):
    """Patch rate_limit_ai to a no-op for all tests to prevent 429 errors."""
    monkeypatch.setattr(
        "backend.route.rate_limit_ai",
        lambda *args, **kwargs: None  # no-op
    )

# Dummy autouse fixture for clearing rate limit store (optional)

@pytest.fixture(autouse=True)
def clear_rate_limit_store():
    """Dummy fixture to satisfy autouse. No Redis clearing is done."""
    yield

# SUCCESS CASE

@patch("backend.route.ai_client.generate")
@patch("backend.route.rate_limit_ai")
def test_process_success(mock_rate_limit, mock_generate):
    mock_generate.return_value = "Processed result"
    response = client.post(
        "/process",
        json={
            "text": "Hello world",
            "feature": FeatureType.summarize.value
        }
    )
    assert response.status_code == 200
    assert response.json()["result"] == "Processed result"
    mock_rate_limit.assert_called_once()

# INPUT VALIDATION FAILURE

def test_empty_input_fails():
    response = client.post(
        "/process",
        json={
            "text": " ",
            "feature": FeatureType.summarize.value
        }
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "empty_input"

# FEATURE CONTRACT FAILURE

def test_generate_questions_requires_word_count():
    response = client.post(
        "/process",
        json={
            "text": "Valid text here",
            "feature": FeatureType.generate_questions.value
        }
    )
    assert response.status_code == 400
    assert "Word count required" in str(response.json()["detail"])

def test_generate_answers_requires_questions():
    response = client.post(
        "/process",
        json={
            "text": "Valid text here",
            "feature": FeatureType.generate_answers.value
        }
    )
    assert response.status_code == 400
    assert "Questions required" in str(response.json()["detail"])

def test_translate_requires_target_language():
    response = client.post(
        "/process",
        json={
            "text": "Valid text here",
            "feature": FeatureType.translate.value
        }
    )
    assert response.status_code == 400
    assert "Target language required" in str(response.json()["detail"])

# RATE LIMIT PASSTHROUGH

@patch("backend.route.rate_limit_ai")
def test_rate_limit_error_propagates(mock_rate_limit):
    mock_rate_limit.side_effect = HTTPException(
        status_code=429,
        detail={"error": "rate_limit_exceeded"}
    )
    response = client.post(
        "/process",
        json={
            "text": "Hello world",
            "feature": FeatureType.summarize.value
        }
    )
    assert response.status_code == 429
    assert response.json()["detail"]["error"] == "rate_limit_exceeded"

# INTERNAL ERROR PROTECTION

@patch("backend.route.ai_client.generate")
@patch("backend.route.rate_limit_ai")
def test_internal_error_returns_500(mock_rate_limit, mock_generate):
    mock_generate.side_effect = Exception("Unexpected failure")
    response = client.post(
        "/process",
        json={
            "text": "Hello world",
            "feature": FeatureType.summarize.value
        }
    )
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "internal_error"
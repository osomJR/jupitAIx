from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, field_validator
from typing import List, Optional
from src.schema import FeatureType
from src.ai_processing import process_with_ai
from src.ai_client import AIClient
from src.ai_validation import validate_text_input
from backend.rate_limit_anonymous_user import rate_limit_ai
from backend.rate_limit_authenticated_user import rate_limit_authenticated_user
from backend.auth0_dependencies import get_current_user_optional

router = APIRouter()
ai_client = AIClient()

# Request / Response Models

class AnalyzerRequest(BaseModel):
    text: str
    feature: FeatureType

    # Optional feature-specific parameters
    
    word_count: Optional[int] = None
    questions: Optional[List[str]] = None
    target_language: Optional[str] = None
    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()
class AIProcessResponse(BaseModel):
    result: str

# Route

@router.post("/process", response_model=AIProcessResponse)
def process_document(
    request: Request,
    payload: AnalyzerRequest,
    auth: dict | None = Depends(get_current_user_optional),
):
    # Step 0 — Rate limit first (cost protection)
    
    if auth:
        rate_limit_authenticated_user(request, auth["user_id"])
    else:
        rate_limit_ai(request, payload.feature)

    # Step 1 — Deterministic input validation
    
    text = validate_text_input(payload.text)
    try:
        # Step 2 — Build prompt using strict contract
        prompt = process_with_ai(
            text=text,
            feature=payload.feature,
            word_count=payload.word_count,
            questions=payload.questions,
            target_language=payload.target_language,
        )

        # Step 3 — Execute AI
        
        output = ai_client.generate(prompt)
        return AIProcessResponse(result=output)
    except HTTPException:
        
        # Preserve structured HTTP errors from lower layers
        
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": str(e),
            },
        )
    except Exception:
        
        # Never leak internal details
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Unexpected processing error.",
            },
        )
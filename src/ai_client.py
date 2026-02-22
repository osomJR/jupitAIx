from fastapi import HTTPException
from openai import OpenAI
import concurrent.futures
from src.validation import (
    validate_structured_text_response,
)
AI_MODEL = "gpt-4o-mini"
MAX_COMPLETION_TOKENS = 1200
AI_TIMEOUT_SECONDS = 45
PROVIDER_TIMEOUT_SECONDS = 30

# OpenAI client with provider-level timeout

client = OpenAI(timeout=PROVIDER_TIMEOUT_SECONDS)
class AIClient:
    """
    Low-level AI execution layer.
    Responsibilities:
    - Send prompts to the AI model
    - Return raw model output
    - Handle provider-level failures
    This layer MUST NOT:
    - Modify prompts
    - Add business rules
    - Decide tone or structure
    """

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=500,
                detail="Empty prompt passed to AI client"
            )

        # Controlled execution thread (timeout isolation)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._call_provider, prompt)

            try:
                result = future.result(timeout=AI_TIMEOUT_SECONDS)

                if not result or not result.strip():
                    raise HTTPException(
                        status_code=502,
                        detail="AI returned empty response"
                    )

                # Deterministic response-level safety check
                # (Structure validation happens upstream depending on feature)
                validate_structured_text_response(result)

                return result

            except concurrent.futures.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "ai_timeout",
                        "message": "AI provider did not respond in time"
                    }
                )

            except HTTPException:
                raise

            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"AI provider error: {str(e)}"
                )

    def _call_provider(self, prompt: str) -> str:
        """
        Isolated provider call.
        Keeps timeout logic clean and testable.
        """

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict document processing AI."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Deterministic output
            max_tokens=MAX_COMPLETION_TOKENS,
        )

        content = response.choices[0].message.content

        if content is None:
            raise HTTPException(
                status_code=502,
                detail="AI provider returned null content"
            )
        return content.strip()

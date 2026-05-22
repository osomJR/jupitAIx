from __future__ import annotations

"""
Centralized API error normalization for the analyzer backend.

What this module does:
- defines stable public error codes
- hides raw internal/provider exception messages from UI users
- maps backend exceptions into friendly FastAPI responses
- installs global exception handlers for FastAPI

Public response shape:
{
    "success": False,
    "error": {
        "code": "SOURCE_FILE_NOT_FOUND",
        "message": "We couldn't find the uploaded file. Please upload it again.",
        "retryable": False
    }
}

Recommended integration:
1) In api_v1.py:
       from backend.errors import install_error_handlers
       ...
       app = FastAPI(...)
       install_error_handlers(app)

2) In route_v1.py, replace the current _bad_request/_service_unavailable flattening with:
       from backend.errors import to_http_exception
       ...
       except Exception as exc:
           raise to_http_exception(exc) from exc

3) Keep raw/internal exception text for logs only. Do not return it to the client.
"""

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    INPUT_REQUIRED = "INPUT_REQUIRED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_ACTION = "INVALID_ACTION"
    INVALID_UPLOAD_METADATA = "INVALID_UPLOAD_METADATA"
    INVALID_FILE_ENCODING = "INVALID_FILE_ENCODING"

    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    UNSUPPORTED_CONVERSION_PAIR = "UNSUPPORTED_CONVERSION_PAIR"
    UNSUPPORTED_OUTPUT_FORMAT = "UNSUPPORTED_OUTPUT_FORMAT"

    FILE_EMPTY = "FILE_EMPTY"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"

    SOURCE_FILE_NOT_FOUND = "SOURCE_FILE_NOT_FOUND"
    UPLOAD_PERSIST_FAILED = "UPLOAD_PERSIST_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    PROCESSING_OUTPUT_MISSING = "PROCESSING_OUTPUT_MISSING"

    WORKFLOW_PREREQUISITE_REQUIRED = "WORKFLOW_PREREQUISITE_REQUIRED"
    FEATURE_NOT_CONFIGURED = "FEATURE_NOT_CONFIGURED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RATE_LIMIT_UNAVAILABLE = "RATE_LIMIT_UNAVAILABLE"

    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    INSUFFICIENT_SCOPE = "INSUFFICIENT_SCOPE"
    AUTH_PROVIDER_UNAVAILABLE = "AUTH_PROVIDER_UNAVAILABLE"

    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_SERVICE_ERROR = "UPSTREAM_SERVICE_ERROR"

    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ErrorDefinition:
    code: ErrorCode
    status_code: int
    error_message: str
    friendly_message: str
    retryable: bool = False


ERRORS: dict[ErrorCode, ErrorDefinition] = {
    ErrorCode.INPUT_REQUIRED: ErrorDefinition(
        code=ErrorCode.INPUT_REQUIRED,
        status_code=400,
        error_message="Required input is missing.",
        friendly_message="Provide the required input and try again.",
    ),
    ErrorCode.INVALID_REQUEST: ErrorDefinition(
        code=ErrorCode.INVALID_REQUEST,
        status_code=422,
        error_message="The request is invalid.",
        friendly_message="Some request fields are invalid. Please review and try again.",
    ),
    ErrorCode.INVALID_ACTION: ErrorDefinition(
        code=ErrorCode.INVALID_ACTION,
        status_code=422,
        error_message="The requested action is invalid.",
        friendly_message="This action is not supported.",
    ),
    ErrorCode.INVALID_UPLOAD_METADATA: ErrorDefinition(
        code=ErrorCode.INVALID_UPLOAD_METADATA,
        status_code=400,
        error_message="The uploaded file metadata is invalid.",
        friendly_message="The uploaded file is missing required information. Please upload it again.",
    ),
    ErrorCode.INVALID_FILE_ENCODING: ErrorDefinition(
        code=ErrorCode.INVALID_FILE_ENCODING,
        status_code=422,
        error_message="The file encoding is invalid.",
        friendly_message="The TXT file uses an unsupported encoding. Save it as UTF-8 and try again.",
    ),
    ErrorCode.UNSUPPORTED_FILE_TYPE: ErrorDefinition(
        code=ErrorCode.UNSUPPORTED_FILE_TYPE,
        status_code=422,
        error_message="The file type is not supported.",
        friendly_message="That file type is not supported for this feature.",
    ),
    ErrorCode.UNSUPPORTED_CONVERSION_PAIR: ErrorDefinition(
        code=ErrorCode.UNSUPPORTED_CONVERSION_PAIR,
        status_code=422,
        error_message="The requested conversion is not supported.",
        friendly_message="That conversion is not supported.",
    ),
    ErrorCode.UNSUPPORTED_OUTPUT_FORMAT: ErrorDefinition(
        code=ErrorCode.UNSUPPORTED_OUTPUT_FORMAT,
        status_code=422,
        error_message="The requested output format is not supported.",
        friendly_message="That output format is not supported.",
    ),
    ErrorCode.FILE_EMPTY: ErrorDefinition(
        code=ErrorCode.FILE_EMPTY,
        status_code=422,
        error_message="The file is empty.",
        friendly_message="The uploaded file is empty.",
    ),
    ErrorCode.FILE_TOO_LARGE: ErrorDefinition(
        code=ErrorCode.FILE_TOO_LARGE,
        status_code=413,
        error_message="The file is too large.",
        friendly_message="The file is too large. Upload a smaller file and try again.",
    ),
    ErrorCode.EXTRACTION_FAILED: ErrorDefinition(
        code=ErrorCode.EXTRACTION_FAILED,
        status_code=422,
        error_message="Text extraction failed.",
        friendly_message="We couldn't read usable text from this file.",
    ),
    ErrorCode.SOURCE_FILE_NOT_FOUND: ErrorDefinition(
        code=ErrorCode.SOURCE_FILE_NOT_FOUND,
        status_code=404,
        error_message="The source file could not be found.",
        friendly_message="We couldn't find the uploaded file. Please upload it again.",
    ),
    ErrorCode.UPLOAD_PERSIST_FAILED: ErrorDefinition(
        code=ErrorCode.UPLOAD_PERSIST_FAILED,
        status_code=500,
        error_message="The uploaded file could not be saved.",
        friendly_message="We couldn't save the uploaded file. Please try again.",
        retryable=True,
    ),
    ErrorCode.PROCESSING_FAILED: ErrorDefinition(
        code=ErrorCode.PROCESSING_FAILED,
        status_code=500,
        error_message="Processing failed.",
        friendly_message="We couldn't complete the request. Please try again.",
        retryable=True,
    ),
    ErrorCode.PROCESSING_OUTPUT_MISSING: ErrorDefinition(
        code=ErrorCode.PROCESSING_OUTPUT_MISSING,
        status_code=500,
        error_message="Processing completed without producing the expected output.",
        friendly_message="Processing finished, but the output file could not be prepared.",
        retryable=True,
    ),
    ErrorCode.WORKFLOW_PREREQUISITE_REQUIRED: ErrorDefinition(
        code=ErrorCode.WORKFLOW_PREREQUISITE_REQUIRED,
        status_code=409,
        error_message="A required workflow prerequisite is missing.",
        friendly_message="Generate questions first, then generate answers.",
    ),
    ErrorCode.FEATURE_NOT_CONFIGURED: ErrorDefinition(
        code=ErrorCode.FEATURE_NOT_CONFIGURED,
        status_code=503,
        error_message="The feature is not configured.",
        friendly_message="This feature is temporarily unavailable.",
        retryable=True,
    ),
    ErrorCode.RATE_LIMIT_EXCEEDED: ErrorDefinition(
        code=ErrorCode.RATE_LIMIT_EXCEEDED,
        status_code=429,
        error_message="Rate limit exceeded.",
        friendly_message="You've reached the usage limit for this feature. Try again later.",
        retryable=True,
    ),
    ErrorCode.RATE_LIMIT_UNAVAILABLE: ErrorDefinition(
        code=ErrorCode.RATE_LIMIT_UNAVAILABLE,
        status_code=503,
        error_message="Rate limiting is unavailable.",
        friendly_message="We can't verify usage limits right now. Please try again later.",
        retryable=True,
    ),
    ErrorCode.AUTHORIZATION_REQUIRED: ErrorDefinition(
        code=ErrorCode.AUTHORIZATION_REQUIRED,
        status_code=401,
        error_message="Authorization is required.",
        friendly_message="Please sign in to continue.",
    ),
    ErrorCode.INVALID_TOKEN: ErrorDefinition(
        code=ErrorCode.INVALID_TOKEN,
        status_code=401,
        error_message="The access token is invalid.",
        friendly_message="Your session is invalid or expired. Please sign in again.",
    ),
    ErrorCode.INSUFFICIENT_SCOPE: ErrorDefinition(
        code=ErrorCode.INSUFFICIENT_SCOPE,
        status_code=403,
        error_message="The token does not have the required scope.",
        friendly_message="You don't have permission to use this feature.",
    ),
    ErrorCode.AUTH_PROVIDER_UNAVAILABLE: ErrorDefinition(
        code=ErrorCode.AUTH_PROVIDER_UNAVAILABLE,
        status_code=503,
        error_message="The authentication provider is unavailable.",
        friendly_message="Sign-in is temporarily unavailable.",
        retryable=True,
    ),
    ErrorCode.UPSTREAM_TIMEOUT: ErrorDefinition(
        code=ErrorCode.UPSTREAM_TIMEOUT,
        status_code=504,
        error_message="An upstream service timed out.",
        friendly_message="The service took too long to respond. Please try again.",
        retryable=True,
    ),
    ErrorCode.UPSTREAM_SERVICE_ERROR: ErrorDefinition(
        code=ErrorCode.UPSTREAM_SERVICE_ERROR,
        status_code=502,
        error_message="An upstream service failed.",
        friendly_message="An external service failed while processing your request. Please try again.",
        retryable=True,
    ),
    ErrorCode.INTERNAL_ERROR: ErrorDefinition(
        code=ErrorCode.INTERNAL_ERROR,
        status_code=500,
        error_message="An internal server error occurred.",
        friendly_message="Something went wrong while processing your request.",
        retryable=False,
    ),
}


@dataclass(frozen=True)
class NormalizedError:
    definition: ErrorDefinition
    internal_message: str
    details: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "success": False,
            "error": {
                "code": self.definition.code.value,
                "message": self.definition.friendly_message,
                "retryable": self.definition.retryable,
            },
        }


class APIError(HTTPException):
    """HTTPException carrying a normalized public error payload."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        internal_message: str | None = None,
        friendly_message: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        definition = ERRORS[code]
        if friendly_message is not None:
            definition = replace(definition, friendly_message=friendly_message)
        if retryable is not None:
            definition = replace(definition, retryable=retryable)

        self.code = code
        self.internal_message = internal_message or definition.error_message
        self.details = details or {}
        self.retryable = definition.retryable
        self.friendly_message = definition.friendly_message

        super().__init__(
            status_code=definition.status_code,
            detail={
                "success": False,
                "error": {
                    "code": definition.code.value,
                    "message": definition.friendly_message,
                    "retryable": definition.retryable,
                },
            },
        )


def raise_api_error(
    code: ErrorCode,
    *,
    internal_message: str | None = None,
    friendly_message: str | None = None,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> APIError:
    return APIError(
        code,
        internal_message=internal_message,
        friendly_message=friendly_message,
        retryable=retryable,
        details=details,
    )


# ---------------------------------------------------------------------------
# Exception -> error-code classification
# ---------------------------------------------------------------------------

def _detail_to_message(detail: Any) -> str:
    if isinstance(detail, Mapping):
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        error = detail.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        return str(detail)
    if isinstance(detail, str):
        return detail.strip()
    return str(detail)


def _class_name(exc: Exception) -> str:
    return exc.__class__.__name__


def _normalized_http_error_from_http_exception(exc: HTTPException | StarletteHTTPException) -> Optional[NormalizedError]:
    detail = exc.detail if hasattr(exc, "detail") else str(exc)
    status_code = getattr(exc, "status_code", 500)
    message = _detail_to_message(detail)
    lowered = message.lower()

    raw_error: str | None = None
    if isinstance(detail, Mapping):
        maybe_error = detail.get("error")
        if isinstance(maybe_error, str):
            raw_error = maybe_error.lower().strip()

    # Already normalized.
    if isinstance(detail, Mapping):
        error_block = detail.get("error")
        if isinstance(error_block, Mapping):
            code = error_block.get("code")
            if isinstance(code, str) and code in ErrorCode.__members__:
                definition = ERRORS[ErrorCode[code]]
                return NormalizedError(definition=definition, internal_message=message)

    # Auth / gateway / provider specific error tags.
    if raw_error == "authorization_required":
        return _norm(ErrorCode.AUTHORIZATION_REQUIRED, message)
    if raw_error == "invalid_token":
        return _norm(ErrorCode.INVALID_TOKEN, message)
    if raw_error == "insufficient_scope":
        return _norm(ErrorCode.INSUFFICIENT_SCOPE, message)
    if raw_error in {"jwks_unavailable", "jwks_invalid"}:
        return _norm(ErrorCode.AUTH_PROVIDER_UNAVAILABLE, message)
    if raw_error in {"ai_timeout", "asr_timeout"} or status_code == 504:
        return _norm(ErrorCode.UPSTREAM_TIMEOUT, message)
    if raw_error == "asr_provider_http_error":
        return _norm(ErrorCode.UPSTREAM_SERVICE_ERROR, message)
    if raw_error == "rate_limit_exceeded" or status_code == 429:
        return _norm(ErrorCode.RATE_LIMIT_EXCEEDED, message)

    # Existing flattened router responses.
    if raw_error in {"invalid_request", "service_unavailable"}:
        return _normalized_error_from_message(message, status_code_hint=status_code)

    # Generic HTTPException with string details from clients/providers.
    if status_code in {401, 403, 429, 502, 503, 504}:
        return _normalized_error_from_message(message, status_code_hint=status_code)

    if status_code == 404 and lowered == "not found":
        return _norm(
            ErrorCode.INVALID_REQUEST,
            "FastAPI route not found.",
            friendly_message="The requested endpoint was not found.",
        )

    return None


def _norm(
    code: ErrorCode,
    internal_message: str,
    *,
    friendly_message: str | None = None,
    retryable: bool | None = None,
) -> NormalizedError:
    definition = ERRORS[code]
    if friendly_message is not None:
        definition = replace(definition, friendly_message=friendly_message)
    if retryable is not None:
        definition = replace(definition, retryable=retryable)
    return NormalizedError(definition=definition, internal_message=internal_message)


def _normalized_error_from_message(message: str, *, status_code_hint: int | None = None) -> NormalizedError:
    lowered = message.lower().strip()

    # Specific / workflow / state errors first.
    if "not a standalone backend action" in lowered or "prior generate_questions completion proof" in lowered:
        return _norm(ErrorCode.WORKFLOW_PREREQUISITE_REQUIRED, message)

    # Authentication and authorization.
    if status_code_hint == 401 or "authorization credentials are required" in lowered:
        if "authorization credentials are required" in lowered:
            return _norm(ErrorCode.AUTHORIZATION_REQUIRED, message)
        return _norm(ErrorCode.INVALID_TOKEN, message)
    if status_code_hint == 403 or "required scopes" in lowered or "insufficient_scope" in lowered:
        return _norm(ErrorCode.INSUFFICIENT_SCOPE, message)

    # Rate limiting.
    if status_code_hint == 429 or "rate limit" in lowered:
        return _norm(ErrorCode.RATE_LIMIT_EXCEEDED, message)
    if "rate limiter" in lowered or "usage limits" in lowered:
        return _norm(ErrorCode.RATE_LIMIT_UNAVAILABLE, message)

    # Missing files.
    if any(token in lowered for token in (
        "source file not found",
        "media file not found",
        "generated artifact not found",
        "uploaded file not found",
        "file not found",
    )):
        return _norm(ErrorCode.SOURCE_FILE_NOT_FOUND, message)

    # Upload / metadata.
    if "failed to persist uploaded file" in lowered:
        return _norm(ErrorCode.UPLOAD_PERSIST_FAILED, message)
    if any(token in lowered for token in (
        "uploaded file must have a filename",
        "uploaded file must include a valid extension",
        "missing its persisted file path",
        "filename is missing",
        "file reference is missing",
        "output filename is missing",
    )):
        return _norm(ErrorCode.INVALID_UPLOAD_METADATA, message)

    # Input required / empty content.
    if any(token in lowered for token in (
        "provide either file or text, not both",
        "either file or text is required",
        "provide either file_path or inline_text, not both",
        "either file_path or inline_text must be provided",
        "no upload file was provided",
        "inline text cannot be empty",
        "empty content cannot be processed",
        "empty content cannot be written",
        "empty prompt passed",
        "empty file_path passed",
    )):
        friendly = "Provide the required input and try again."
        if "not both" in lowered:
            friendly = "Provide one input source only and try again."
        elif "either file or text is required" in lowered or "either file_path or inline_text must be provided" in lowered:
            friendly = "Add a file or enter text to continue."
        elif "no upload file was provided" in lowered:
            friendly = "Please choose a file to upload."
        elif "inline text cannot be empty" in lowered or "empty content" in lowered:
            friendly = "Enter some text to continue."
        return _norm(ErrorCode.INPUT_REQUIRED, message, friendly_message=friendly)

    # File size / emptiness / encoding.
    if "file is empty" in lowered:
        return _norm(ErrorCode.FILE_EMPTY, message)
    if "exceeds maximum allowed size" in lowered or "file is too large" in lowered:
        return _norm(ErrorCode.FILE_TOO_LARGE, message)
    if "utf-8" in lowered and "txt file" in lowered:
        return _norm(ErrorCode.INVALID_FILE_ENCODING, message)

    # Extraction / OCR failures.
    if any(token in lowered for token in (
        "document text could not be extracted",
        "document contains no words",
        "contains no pages",
    )):
        return _norm(ErrorCode.EXTRACTION_FAILED, message)
    if "ocr language token cannot be empty" in lowered:
        return _norm(ErrorCode.INVALID_REQUEST, message, friendly_message="One of the OCR language settings is invalid.")
    if any(token in lowered for token in (
        "traineddata is not installed",
        "no ocr languages are installed",
    )):
        return _norm(ErrorCode.FEATURE_NOT_CONFIGURED, message, friendly_message="OCR is temporarily unavailable.")

    # Unsupported inputs / formats / conversions.
    if "unsupported conversion pair" in lowered:
        return _norm(ErrorCode.UNSUPPORTED_CONVERSION_PAIR, message)
    if any(token in lowered for token in (
        "unsupported writer output format",
        "output_format must be one of: pdf, docx",
    )):
        return _norm(ErrorCode.UNSUPPORTED_OUTPUT_FORMAT, message)
    if any(token in lowered for token in (
        "unsupported file extension",
        "unsupported file format",
        "audio uploads must be mp3",
        "video uploads must be one of",
        "only supports input formats",
        "convert only supports",
        "redact/data_mask only support",
        "unsupported redact input format",
        "unsupported data_mask input format",
        "must be one of: mp3, mp4, mkv, mov, wav",
        "format must be one of: pdf, docx, jpg, jpeg, png",
    )):
        return _norm(ErrorCode.UNSUPPORTED_FILE_TYPE, message)

    # Invalid action / contract mismatch.
    if any(token in lowered for token in (
        "unsupported action",
        "unsupported document upload action",
        "unsupported document action for extraction",
        "requires action='redact'",
        "requires action='data_mask'",
        "only supports action='redact' or action='data_mask'",
    )):
        return _norm(ErrorCode.INVALID_ACTION, message)

    # Dependencies / setup / missing infra.
    if any(token in lowered for token in (
        "not configured",
        "set google_sdp_project_id",
        "client library is required",
        "not installed",
        "not found on path",
        "project-specific rate limiter wiring remains unchanged",
        "category must be either 'documents' or 'media'",
    )):
        # Special rate-limiter path.
        if "rate limiter" in lowered:
            return _norm(ErrorCode.RATE_LIMIT_UNAVAILABLE, message)
        return _norm(ErrorCode.FEATURE_NOT_CONFIGURED, message)

    # Output missing after successful processing.
    if any(token in lowered for token in (
        "completed without producing an output file",
        "completed without producing the expected output",
        "expected output file was not found",
        "completed but the expected output file was not found",
        "without producing a pdf output file",
        "document writing completed without producing an output file",
        "audio extraction completed without producing an output file",
    )):
        return _norm(ErrorCode.PROCESSING_OUTPUT_MISSING, message)

    # Processing runtime failures.
    if any(token in lowered for token in (
        "failed while converting",
        "failed while extracting audio",
        "provider error",
        "provider returned empty output",
        "provider returned null or unreadable",
        "transcription service returned an error",
        "provider returned an unreadable result",
    )):
        if "provider" in lowered or status_code_hint in {502, 504}:
            return _norm(ErrorCode.UPSTREAM_SERVICE_ERROR, message)
        return _norm(ErrorCode.PROCESSING_FAILED, message)

    # Upstream status hints.
    if status_code_hint == 504:
        return _norm(ErrorCode.UPSTREAM_TIMEOUT, message)
    if status_code_hint == 502:
        return _norm(ErrorCode.UPSTREAM_SERVICE_ERROR, message)
    if status_code_hint == 503:
        return _norm(ErrorCode.FEATURE_NOT_CONFIGURED, message)

    # Remaining type/value/contract errors.
    if any(token in lowered for token in (
        "must be a string",
        "cannot be empty",
        "requires",
        "must be >=",
        "must be <=",
        "must be one of",
        "payload",
        "detected_language must not be provided",
        "questions list cannot be empty",
        "questions must be sequentially numbered",
        "answer count must exactly match",
        "structure_preservation must be",
        "output must be",
        "response must be",
        "invalid",
    )):
        return _norm(ErrorCode.INVALID_REQUEST, message)

    return _norm(ErrorCode.INTERNAL_ERROR, message)


def normalize_exception(exc: Exception) -> NormalizedError:
    """Map any backend exception to a normalized public error."""
    if isinstance(exc, APIError):
        definition = ERRORS[exc.code]
        return NormalizedError(definition=definition, internal_message=exc.internal_message, details=exc.details)

    if isinstance(exc, RequestValidationError):
        return _norm(ErrorCode.INVALID_REQUEST, "FastAPI request validation failed.")

    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        normalized = _normalized_http_error_from_http_exception(exc)
        if normalized is not None:
            return normalized
        return _normalized_error_from_message(_detail_to_message(exc.detail), status_code_hint=exc.status_code)  # type: ignore[arg-type]

    if isinstance(exc, FileNotFoundError):
        return _norm(ErrorCode.SOURCE_FILE_NOT_FOUND, str(exc))

    if isinstance(exc, NotImplementedError):
        return _norm(ErrorCode.RATE_LIMIT_UNAVAILABLE, str(exc) or "Required infrastructure is not wired.")

    if _class_name(exc) == "UploadError":
        return _normalized_error_from_message(str(exc), status_code_hint=400)

    if isinstance(exc, RuntimeError):
        return _normalized_error_from_message(str(exc), status_code_hint=500)

    if isinstance(exc, (ValueError, TypeError)):
        return _normalized_error_from_message(str(exc), status_code_hint=422)

    return _norm(ErrorCode.INTERNAL_ERROR, str(exc) or exc.__class__.__name__)


def to_http_exception(exc: Exception) -> APIError:
    normalized = normalize_exception(exc)
    return APIError(
        normalized.definition.code,
        internal_message=normalized.internal_message,
        details=normalized.details,
    )

# FastAPI integration

def _log_normalized_error(request: Request, normalized: NormalizedError, exc: Exception) -> None:
    level = logging.WARNING
    if normalized.definition.status_code >= 500:
        level = logging.ERROR

    logger.log(
        level,
        "Handled API error",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error_code": normalized.definition.code.value,
            "status_code": normalized.definition.status_code,
            "internal_message": normalized.internal_message,
        },
        exc_info=exc if normalized.definition.status_code >= 500 else None,
    )


async def _json_error_response(request: Request, exc: Exception) -> JSONResponse:
    normalized = normalize_exception(exc)
    _log_normalized_error(request, normalized, exc)
    return JSONResponse(
        status_code=normalized.definition.status_code,
        content=normalized.payload(),
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return await _json_error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return await _json_error_response(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return await _json_error_response(request, exc)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return await _json_error_response(request, exc)



__all__ = [
    "APIError",
    "ERRORS",
    "ErrorCode",
    "ErrorDefinition",
    "NormalizedError",
    "install_error_handlers",
    "normalize_exception",
    "raise_api_error",
    "to_http_exception",
]

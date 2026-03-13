from __future__ import annotations

from typing import Any, Mapping, Optional, Type, Union

from pydantic import BaseModel

from .schema import (
    # Core request/response envelope
    AnalyzerRequest,
    AnalyzerResponse,
    FeatureType,
    # Inputs
    DocumentPayload,
    MediaPayload,
    DocumentInputFormat,
    MediaType,
    # Payloads
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    TranscriptionRequest,
    ExplanationRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
    # Results + metadata
    DeterminismMetadata,
    InlineTextResult,
    FileResult,
    FileOutputFormat,
    QuestionScale,
    QuestionScaleMetadata,
    QuestionGenerationInlineResult,
    QuestionGenerationFileResult,
    AnswerGenerationInlineResult,
    AnswerGenerationFileResult,
    classify_word_count,
    # Limits
    MAX_WORD_COUNT
)

# =========================
# Core request validation
# =========================

_ACTION_PAYLOAD_MAP: dict[FeatureType, Type[BaseModel]] = {
    FeatureType.convert: ConversionRequest,
    FeatureType.summarize: SummarizationRequest,
    FeatureType.grammar_correct: GrammarCorrectionRequest,
    FeatureType.translate: TranslationRequest,
    FeatureType.transcribe: TranscriptionRequest,
    FeatureType.explain: ExplanationRequest,
    FeatureType.generate_questions: QuestionGenerationRequest,
    FeatureType.generate_answers: AnswerGenerationRequest,
}

def validate_action_payload_consistency(request: AnalyzerRequest) -> None:
    """
    Ensures request.payload is the correct model for request.action.
    Note: schema already enforces payload.feature == action; this is an extra type guard.
    """
    if request.payload is None:
        raise ValueError("payload is required")

    expected_model = _ACTION_PAYLOAD_MAP.get(request.action)
    if expected_model is None:
        raise ValueError(f"Unsupported action: {request.action}")

    if not isinstance(request.payload, expected_model):
        raise ValueError(
            f"payload type mismatch for action '{request.action.value}': "
            f"expected {expected_model.__name__}, got {type(request.payload).__name__}"
        )

def validate_no_client_detected_language(request: AnalyzerRequest) -> None:
    """
    Mirrors schema contract: client must not supply detected_language; server supplies detection.
    (Schema already enforces this, but this makes validation.py strictly aligned/explicit.)
    """
    if isinstance(request.input, DocumentPayload):
        if request.input.metadata.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection.")
    if isinstance(request.input, MediaPayload):
        if request.input.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection.")

def validate_word_count_bounds_when_present(request: AnalyzerRequest) -> None:
    """
    Defensive guard around extracted_word_count.
    - Schema enforces: extracted_word_count is Optional[int] with ge=0, le=MAX_WORD_COUNT.
    - For AI doc actions, schema's AnalyzerRequest validator further enforces it's present and >= 1.
    """
    if isinstance(request.input, DocumentPayload):
        wc = request.input.metadata.extracted_word_count
        if wc is None:
            return
        if wc < 0:
            raise ValueError("extracted_word_count must be >= 0.")
        if wc > MAX_WORD_COUNT:
            raise ValueError(f"extracted_word_count must be <= {MAX_WORD_COUNT}.")

def validate_question_scale(word_count: int) -> QuestionScale:
    """
    Deterministically classifies question scale based on schema scaling rules.
    """
    return classify_word_count(word_count).classification

def get_question_range(word_count: int) -> tuple[int, int]:
    """
    Returns (min_questions, max_questions) based on schema scaling rules.
    """
    rule = classify_word_count(word_count)
    return rule.min_questions, rule.max_questions

def validate_analyzer_request(
    request: Union[AnalyzerRequest, Mapping[str, Any]]
) -> AnalyzerRequest:
    """
    Full deterministic validation pipeline for incoming requests.

    Primary contract enforcement lives in schema.py via Pydantic validators.
    This function adds:
      - explicit payload type guard
      - explicit guard against client-supplied detected_language
      - defensive extracted_word_count bounds check when present
    """

    req = request if isinstance(request, AnalyzerRequest) else AnalyzerRequest.model_validate(request)

    validate_action_payload_consistency(req)
    validate_no_client_detected_language(req)
    validate_word_count_bounds_when_present(req)

    # Optional explicit checks (schema already enforces the critical ones)
    if req.action == FeatureType.generate_questions:
        if not isinstance(req.input, DocumentPayload):
            raise ValueError("generate_questions requires DocumentPayload input.")
        if req.input.metadata.extracted_word_count is None:
            raise ValueError("generate_questions requires extracted_word_count.")
        validate_question_scale(req.input.metadata.extracted_word_count)

    if req.action == FeatureType.generate_answers:
        if not isinstance(req.payload, AnswerGenerationRequest):
            raise ValueError("generate_answers requires AnswerGenerationRequest payload.")
        if not req.payload.questions:
            raise ValueError("Questions list cannot be empty.")

    return req

# =========================
# Result construction helpers (aligned to schema.py)
# =========================

def _meta(*, algorithm_version: Optional[str] = None) -> DeterminismMetadata:
    return DeterminismMetadata(algorithm_version=algorithm_version)

def build_inline_txt_result(
    *,
    content: str,
    algorithm_version: Optional[str] = None,
) -> InlineTextResult:
    return InlineTextResult(content=content, meta=_meta(algorithm_version=algorithm_version))

def build_file_result(
    *,
    filename: str,
    output_format: FileOutputFormat,
    file_size_mb: float,
    algorithm_version: Optional[str] = None,
) -> FileResult:
    return FileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        meta=_meta(algorithm_version=algorithm_version),
    )

def build_question_generation_inline_result(
    *,
    content: str,
    extracted_word_count: int,
    algorithm_version: Optional[str] = None,
) -> QuestionGenerationInlineResult:
    classification = classify_word_count(extracted_word_count).classification
    return QuestionGenerationInlineResult(
        content=content,
        meta=_meta(algorithm_version=algorithm_version),
        scale=QuestionScaleMetadata(
            classification=classification,
            extracted_word_count=extracted_word_count,
        ),
    )

def build_question_generation_file_result(
    *,
    filename: str,
    output_format: FileOutputFormat,
    file_size_mb: float,
    extracted_word_count: int,
    algorithm_version: Optional[str] = None,
) -> QuestionGenerationFileResult:
    classification = classify_word_count(extracted_word_count).classification
    return QuestionGenerationFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        meta=_meta(algorithm_version=algorithm_version),
        scale=QuestionScaleMetadata(
            classification=classification,
            extracted_word_count=extracted_word_count,
        ),
    )

def build_answer_generation_inline_result(
    *,
    content: str,
    expected_question_count: int,
    algorithm_version: Optional[str] = None,
) -> AnswerGenerationInlineResult:
    return AnswerGenerationInlineResult(
        content=content,
        meta=_meta(algorithm_version=algorithm_version),
        expected_question_count=expected_question_count,
    )

def build_answer_generation_file_result(
    *,
    filename: str,
    output_format: FileOutputFormat,
    file_size_mb: float,
    expected_question_count: int,
    algorithm_version: Optional[str] = None,
) -> AnswerGenerationFileResult:
    return AnswerGenerationFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        meta=_meta(algorithm_version=algorithm_version),
        expected_question_count=expected_question_count,
    )

# =========================
# Response validation
# =========================

def _expected_response_input_format(request: AnalyzerRequest):
    """
    AnalyzerResponse echoes input_format as:
      - DocumentInputFormat for doc requests
      - "audio" or "video" for transcription requests
    """
    if isinstance(request.input, MediaPayload):
        return "audio" if request.input.media_type == MediaType.audio else "video"
    return request.input.metadata.input_format

def validate_analyzer_response(
    response: Union[AnalyzerResponse, Mapping[str, Any]],
    *,
    request: Optional[AnalyzerRequest] = None,
) -> AnalyzerResponse:
    """
    Validates that the response matches schema rules.

    Schema enforces:
      - action->result family constraints
      - output-extension rules for AI doc actions
      - transcription output rule (always inline txt)
      - language boundary validation when language fields are provided

    This function adds:
      - if request is provided, response.action/policy/input_format/ui_language/system_language must match it.
    """
    resp = response if isinstance(response, AnalyzerResponse) else AnalyzerResponse.model_validate(response)

    if request is not None:
        if resp.action != request.action:
            raise ValueError("Response action must match request action.")
        if resp.policy != request.policy:
            raise ValueError("Response policy must match request policy.")

        # Mirror request envelope echo fields
        if resp.ui_language != request.ui_language:
            raise ValueError("Response ui_language must match request ui_language.")
        if resp.system_language != request.system_language:
            raise ValueError("Response system_language must match request system_language.")

        expected_in_fmt = _expected_response_input_format(request)

        # AnalyzerResponse.input_format is Union[DocumentInputFormat, Literal['audio'], Literal['video']]
        if isinstance(expected_in_fmt, DocumentInputFormat):
            if resp.input_format != expected_in_fmt:
                raise ValueError("Response input_format must match request document input_format.")
        else:
            if resp.input_format != expected_in_fmt:
                raise ValueError("Response input_format must match request media type (audio/video).")

    return resp
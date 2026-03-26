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
    MediaType,
    # Payloads
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    TranscriptionRequest,
    ExplanationRequest,
    RedactionRequest,
    DataMaskingRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
    # Policies
    OutputPolicy,
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
    MAX_WORD_COUNT,
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
    FeatureType.redact: RedactionRequest,
    FeatureType.data_mask: DataMaskingRequest,
    FeatureType.generate_questions: QuestionGenerationRequest,
    FeatureType.generate_answers: AnswerGenerationRequest,
}

# Strict backend gate for generate_answers:
# The request must prove that generate_questions completed first.
# This is intentionally implemented in validation.py because the current schema.py
# does not model workflow state as a first-class request field.
GENERATE_ANSWERS_WORKFLOW_FIELD = "workflow"
GENERATE_ANSWERS_COMPLETION_FLAG = "generate_questions_completed"
GENERATE_ANSWERS_TOKEN_FIELD = "generate_questions_token"


def _has_nonempty_token(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_generate_answers_prerequisite(
    request: Union[AnalyzerRequest, Mapping[str, Any]],
    *,
    generate_questions_completed: bool = False,
    generate_questions_token: Optional[str] = None,
) -> None:
    """
    Enforces the product rule that generate_answers is not a standalone backend action.

    Accepted proof of prior completion:
    - a non-empty top-level generate_questions_token, OR
    - workflow.generate_questions_completed == True, OR
    - function-level override parameters for already-instantiated AnalyzerRequest objects

    Why this lives here:
    - current schema.py does not define workflow/token fields, so raw transport-level
      metadata must be checked before/alongside schema validation.
    """
    action_value: Optional[str] = None
    raw_token: Optional[str] = None
    raw_completed = False

    if isinstance(request, AnalyzerRequest):
        action_value = request.action.value
    else:
        action_raw = request.get("action")
        action_value = action_raw.value if hasattr(action_raw, "value") else action_raw

        token_value = request.get(GENERATE_ANSWERS_TOKEN_FIELD)
        if _has_nonempty_token(token_value):
            raw_token = token_value

        workflow = request.get(GENERATE_ANSWERS_WORKFLOW_FIELD)
        if isinstance(workflow, Mapping):
            raw_completed = workflow.get(GENERATE_ANSWERS_COMPLETION_FLAG) is True

    if action_value != FeatureType.generate_answers.value:
        return

    has_token = _has_nonempty_token(generate_questions_token) or _has_nonempty_token(raw_token)
    has_completion_marker = generate_questions_completed or raw_completed

    if not (has_token or has_completion_marker):
        raise ValueError(
            "generate_answers is not a standalone backend action. "
            "A prior generate_questions completion proof is required: "
            "provide either a non-empty generate_questions_token or "
            "workflow.generate_questions_completed=True."
        )


def validate_action_payload_consistency(request: AnalyzerRequest) -> None:
    """
    Extra explicit type guard.

    schema.py already enforces payload.feature == action. This function makes the
    expected payload model for each action explicit for downstream callers.
    """
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
    Mirrors the schema contract: detected_language is server-supplied only.
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

    schema.py already enforces bounds, but this keeps validation.py explicit and
    safe for any pre-instantiated or partially transformed request objects.
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
    Returns (min_questions, max_questions) from the schema scaling rules.
    """
    rule = classify_word_count(word_count)
    return rule.min_questions, rule.max_questions


def validate_generate_answers_request(request: AnalyzerRequest) -> None:
    """
    Strict answer-generation checks that align with the non-standalone product flow.

    Backend validation verifies that:
    - prior generate_questions completion proof exists
    - questions were supplied
    - the count of supplied questions is compatible with the document size
    """
    if request.action != FeatureType.generate_answers:
        return

    if not isinstance(request.input, DocumentPayload):
        raise ValueError("generate_answers requires DocumentPayload input.")

    if not isinstance(request.payload, AnswerGenerationRequest):
        raise ValueError("generate_answers requires AnswerGenerationRequest payload.")

    wc = request.input.metadata.extracted_word_count
    if wc is None:
        raise ValueError("generate_answers requires extracted_word_count.")

    rule = classify_word_count(wc)
    question_count = len(request.payload.questions)
    if not (rule.min_questions <= question_count <= rule.max_questions):
        raise ValueError(
            "Supplied questions count is inconsistent with document-size scaling for generate_answers: "
            f"expected {rule.min_questions}–{rule.max_questions}, got {question_count}."
        )


def validate_analyzer_request(
    request: Union[AnalyzerRequest, Mapping[str, Any]],
    *,
    generate_questions_completed: bool = False,
    generate_questions_token: Optional[str] = None,
) -> AnalyzerRequest:
    """
    Full deterministic validation pipeline for incoming requests.

    Primary contract enforcement lives in schema.py via Pydantic validators.
    This function adds explicit alignment checks plus a strict workflow gate
    for generate_answers so downstream code can reject standalone answer-generation calls.
    """
    validate_generate_answers_prerequisite(
        request,
        generate_questions_completed=generate_questions_completed,
        generate_questions_token=generate_questions_token,
    )

    req = request if isinstance(request, AnalyzerRequest) else AnalyzerRequest.model_validate(request)

    validate_action_payload_consistency(req)
    validate_no_client_detected_language(req)
    validate_word_count_bounds_when_present(req)

    if req.action == FeatureType.generate_questions:
        if not isinstance(req.input, DocumentPayload):
            raise ValueError("generate_questions requires DocumentPayload input.")
        if req.input.metadata.extracted_word_count is None:
            raise ValueError("generate_questions requires extracted_word_count.")
        validate_question_scale(req.input.metadata.extracted_word_count)

    if req.action == FeatureType.generate_answers:
        validate_generate_answers_request(req)

    if req.action in {
        FeatureType.convert,
        FeatureType.summarize,
        FeatureType.grammar_correct,
        FeatureType.translate,
        FeatureType.transcribe,
        FeatureType.explain,
        FeatureType.redact,
        FeatureType.data_mask,
        FeatureType.generate_questions,
        FeatureType.generate_answers,
    }:
        if not isinstance(req.policy, OutputPolicy):
            raise ValueError(f"{req.action.value} requires OutputPolicy.")

    return req


# =========================
# Result construction helpers
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
      - DocumentInputFormat for document requests
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
    Validates that the response matches schema rules and, when request is supplied,
    matches the specific request shape.
    """
    resp = response if isinstance(response, AnalyzerResponse) else AnalyzerResponse.model_validate(response)

    if request is not None:
        if resp.action != request.action:
            raise ValueError("Response action must match request action.")
        if resp.policy != request.policy:
            raise ValueError("Response policy must match request policy.")
        if resp.system_language != request.system_language:
            raise ValueError("Response system_language must match request system_language.")

        expected_in_fmt = _expected_response_input_format(request)
        if resp.input_format != expected_in_fmt:
            raise ValueError("Response input_format must match request input format.")

        if request.action == FeatureType.translate:
            assert isinstance(request.payload, TranslationRequest)
            if resp.output_language is not None and resp.output_language != request.payload.target_language:
                raise ValueError("For translate, output_language must match payload.target_language.")

        if request.action == FeatureType.generate_questions:
            if not isinstance(resp.result, (QuestionGenerationInlineResult, QuestionGenerationFileResult)):
                raise ValueError("generate_questions response must contain a question-generation result.")

            if isinstance(request.input, DocumentPayload):
                wc = request.input.metadata.extracted_word_count
                if wc is None:
                    raise ValueError("generate_questions requires extracted_word_count in request input.")
                if resp.result.scale.extracted_word_count != wc:
                    raise ValueError(
                        "Question-generation response scale.extracted_word_count must match request extracted_word_count."
                    )
                expected_classification = classify_word_count(wc).classification
                if resp.result.scale.classification != expected_classification:
                    raise ValueError(
                        "Question-generation response scale.classification is inconsistent with request extracted_word_count."
                    )

        if request.action == FeatureType.generate_answers:
            assert isinstance(request.payload, AnswerGenerationRequest)
            if not isinstance(resp.result, (AnswerGenerationInlineResult, AnswerGenerationFileResult)):
                raise ValueError("generate_answers response must contain an answer-generation result.")
            expected_count = len(request.payload.questions)
            if resp.result.expected_question_count != expected_count:
                raise ValueError("expected_question_count must match the supplied questions count.")

    return resp


__all__ = [
    "GENERATE_ANSWERS_WORKFLOW_FIELD",
    "GENERATE_ANSWERS_COMPLETION_FLAG",
    "GENERATE_ANSWERS_TOKEN_FIELD",
    "validate_generate_answers_prerequisite",
    "validate_action_payload_consistency",
    "validate_no_client_detected_language",
    "validate_word_count_bounds_when_present",
    "validate_question_scale",
    "get_question_range",
    "validate_generate_answers_request",
    "validate_analyzer_request",
    "build_inline_txt_result",
    "build_file_result",
    "build_question_generation_inline_result",
    "build_question_generation_file_result",
    "build_answer_generation_inline_result",
    "build_answer_generation_file_result",
    "validate_analyzer_response",
]

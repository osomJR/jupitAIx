from __future__ import annotations

import pytest

from pydantic import ValidationError

from src.schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    FeatureType,
    OutputPolicy,
    DocumentPayload,
    DocumentMetadata,
    MediaPayload,
    DocumentInputFormat,
    MediaType,
    AudioFormat,
    # Payloads
    ConversionRequest,
    ConversionOutputFormat,
    SummarizationRequest,
    TranslationRequest,
    TranscriptionRequest,
    # Results
    InlineTextResult,
    FileOutputFormat,
    QuestionGenerationInlineResult,
    AnswerGenerationInlineResult,
    # Limits
    FREE_USER_MAX_ACTIONS_PER_DAY,
    MAX_WORD_COUNT,
)

from src.validation import (
    validate_usage_limits,
    validate_action_payload_consistency,
    validate_no_client_detected_language,
    validate_word_count_bounds_when_present,
    validate_question_scale,
    get_question_range,
    validate_analyzer_request,
    validate_analyzer_response,
    build_inline_txt_result,
    build_file_result,
    build_question_generation_inline_result,
    build_answer_generation_inline_result,
)


# -------------------------
# Helpers
# -------------------------

def _policy_for(action: FeatureType) -> OutputPolicy:
    transformed = {
        FeatureType.convert,
        FeatureType.summarize,
        FeatureType.grammar_correct,
        FeatureType.translate,
        FeatureType.transcribe,
    }
    return OutputPolicy(structure_preservation=(action in transformed))


def _doc_input_valid(
    *,
    fmt: DocumentInputFormat,
    text: str | None,
    extracted_word_count: int | None,
) -> DocumentPayload:
    return DocumentPayload(
        text=text,
        metadata=DocumentMetadata(
            input_format=fmt,
            file_size_mb=0.5,
            extracted_word_count=extracted_word_count,
            ocr_used=False,
        ),
    )


def _media_input_audio_valid() -> MediaPayload:
    return MediaPayload(
        media_type=MediaType.audio,
        media_format=AudioFormat.mp3,
        file_size_mb=1.0,
        duration_seconds=10,
        filename="clip.mp3",
    )


def _doc_input_construct(
    *,
    fmt: DocumentInputFormat,
    text: str | None,
    extracted_word_count: int | None,
    detected_language: str | None = None,
) -> DocumentPayload:
    """
    Bypasses schema validators by using model_construct() on DocumentMetadata/DocumentPayload.
    Use ONLY for unit-testing validation.py guards that otherwise would be blocked by schema.
    """
    md = DocumentMetadata.model_construct(
        input_format=fmt,
        file_size_mb=0.5,
        extracted_word_count=extracted_word_count,
        ocr_used=False,
        detected_language=detected_language,
    )
    return DocumentPayload.model_construct(text=text, metadata=md)


def _media_input_audio_construct(*, detected_language: str | None = None) -> MediaPayload:
    return MediaPayload.model_construct(
        media_type=MediaType.audio,
        media_format=AudioFormat.mp3,
        file_size_mb=1.0,
        duration_seconds=10,
        filename="clip.mp3",
        detected_language=detected_language,
    )


# -------------------------
# validate_usage_limits
# -------------------------

def test_validate_usage_limits_noop_when_none():
    validate_usage_limits(None)


def test_validate_usage_limits_rejects_at_or_above_limit_mapping():
    usage = {"is_free_user": True, "actions_today": FREE_USER_MAX_ACTIONS_PER_DAY}
    with pytest.raises(ValueError, match="Daily action limit reached"):
        validate_usage_limits(usage)


def test_validate_usage_limits_invalid_actions_today_type():
    usage = {"is_free_user": True, "actions_today": "not-an-int"}
    with pytest.raises(ValueError, match="Invalid usage_snapshot.actions_today"):
        validate_usage_limits(usage)


# -------------------------
# Core validators (unit-level)
# -------------------------

def test_validate_action_payload_consistency_rejects_wrong_payload_type():
    # Can't instantiate invalid AnalyzerRequest normally (schema blocks it),
    # so we bypass with model_construct().
    req = AnalyzerRequest.model_construct(
        action=FeatureType.summarize,
        input=_doc_input_valid(fmt=DocumentInputFormat.txt, text="hello", extracted_word_count=1),
        payload=TranslationRequest.model_construct(feature=FeatureType.translate, target_language="es"),
        policy=_policy_for(FeatureType.summarize),
        ui_language=None,
        system_language=None,
    )
    with pytest.raises(ValueError, match="payload type mismatch"):
        validate_action_payload_consistency(req)


def test_validate_no_client_detected_language_rejects_document_detected_language():
    req = AnalyzerRequest.model_construct(
        action=FeatureType.summarize,
        input=_doc_input_construct(
            fmt=DocumentInputFormat.txt,
            text="hello",
            extracted_word_count=1,
            detected_language="en",
        ),
        payload=SummarizationRequest.model_construct(feature=FeatureType.summarize),
        policy=_policy_for(FeatureType.summarize),
        ui_language=None,
        system_language=None,
    )
    with pytest.raises(ValueError, match="detected_language must not be provided"):
        validate_no_client_detected_language(req)


def test_validate_no_client_detected_language_rejects_media_detected_language():
    req = AnalyzerRequest.model_construct(
        action=FeatureType.transcribe,
        input=_media_input_audio_construct(detected_language="en"),
        payload=TranscriptionRequest.model_construct(feature=FeatureType.transcribe),
        policy=_policy_for(FeatureType.transcribe),
        ui_language=None,
        system_language=None,
    )
    with pytest.raises(ValueError, match="detected_language must not be provided"):
        validate_no_client_detected_language(req)


def test_validate_word_count_bounds_when_present_rejects_negative():
    # DocumentMetadata normally enforces ge=0; bypass to unit-test guard.
    req = AnalyzerRequest.model_construct(
        action=FeatureType.convert,
        input=_doc_input_construct(fmt=DocumentInputFormat.pdf, text=None, extracted_word_count=-1),
        payload=ConversionRequest.model_construct(
            feature=FeatureType.convert, output_format=ConversionOutputFormat.docx
        ),
        policy=_policy_for(FeatureType.convert),
        ui_language=None,
        system_language=None,
    )
    with pytest.raises(ValueError, match="extracted_word_count must be >= 0"):
        validate_word_count_bounds_when_present(req)


def test_validate_word_count_bounds_when_present_rejects_too_large():
    req = AnalyzerRequest.model_construct(
        action=FeatureType.convert,
        input=_doc_input_construct(
            fmt=DocumentInputFormat.pdf, text=None, extracted_word_count=MAX_WORD_COUNT + 1
        ),
        payload=ConversionRequest.model_construct(
            feature=FeatureType.convert, output_format=ConversionOutputFormat.docx
        ),
        policy=_policy_for(FeatureType.convert),
        ui_language=None,
        system_language=None,
    )
    with pytest.raises(ValueError, match=r"extracted_word_count must be <= "):
        validate_word_count_bounds_when_present(req)


# -------------------------
# Question scaling helpers
# -------------------------

def test_question_scale_helpers_are_consistent():
    wc = 60
    scale = validate_question_scale(wc)
    mn, mx = get_question_range(wc)
    assert scale is not None
    assert isinstance(mn, int) and isinstance(mx, int)
    assert 1 <= mn <= mx


# -------------------------
# validate_analyzer_request (integration-level)
# -------------------------

def test_validate_analyzer_request_accepts_valid_summarize_request():
    req = {
        "action": "summarize",
        "input": {
            "text": "Hello world",
            "metadata": {"input_format": "txt", "file_size_mb": 0.5, "extracted_word_count": 2},
        },
        "payload": {"feature": "summarize"},
        "policy": {"tone_preservation": True, "professional_neutrality": True, "structure_preservation": True},
    }
    out = validate_analyzer_request(req)
    assert isinstance(out, AnalyzerRequest)
    assert out.action == FeatureType.summarize


def test_validate_analyzer_request_enforces_usage_limit_when_snapshot_provided():
    req = AnalyzerRequest(
        action=FeatureType.convert,
        input=_doc_input_valid(fmt=DocumentInputFormat.pdf, text=None, extracted_word_count=None),
        payload=ConversionRequest(feature=FeatureType.convert, output_format=ConversionOutputFormat.docx),
        policy=_policy_for(FeatureType.convert),
    )
    usage = {"is_free_user": True, "actions_today": FREE_USER_MAX_ACTIONS_PER_DAY}
    with pytest.raises(ValueError, match="Daily action limit reached"):
        validate_analyzer_request(req, usage_snapshot=usage)


def test_validate_analyzer_request_generate_answers_rejects_empty_questions():
    req_dict = {
        "action": "generate_answers",
        "input": {
            "text": "Some text",
            "metadata": {"input_format": "txt", "file_size_mb": 0.5, "extracted_word_count": 2},
        },
        "payload": {"feature": "generate_answers", "allow_external_knowledge": False, "questions": []},
        "policy": {"tone_preservation": True, "professional_neutrality": True, "structure_preservation": False},
    }
    with pytest.raises(ValueError, match="Questions list cannot be empty"):
        validate_analyzer_request(req_dict)


def test_validate_analyzer_request_generate_questions_requires_doc_payload_and_wordcount():
    req_dict = {
        "action": "generate_questions",
        "input": {
            "text": "Some text",
            "metadata": {"input_format": "txt", "file_size_mb": 0.5, "extracted_word_count": 10},
        },
        "payload": {"feature": "generate_questions", "allow_external_knowledge": False},
        "policy": {"tone_preservation": True, "professional_neutrality": True, "structure_preservation": False},
    }
    out = validate_analyzer_request(req_dict)
    assert out.action == FeatureType.generate_questions
    assert isinstance(out.input, DocumentPayload)


# -------------------------
# Result builders
# -------------------------

def test_build_inline_txt_result_creates_inline_result():
    r = build_inline_txt_result(content="ok", algorithm_version="v1")
    assert isinstance(r, InlineTextResult)
    assert r.content == "ok"


def test_build_file_result_creates_file_result():
    r = build_file_result(
        filename="out.pdf",
        output_format=FileOutputFormat.pdf,
        file_size_mb=1.2,
        algorithm_version="v1",
    )
    assert r.filename == "out.pdf"
    assert r.output_format == FileOutputFormat.pdf


def test_build_question_generation_inline_result_includes_scale_metadata():
    # Must satisfy schema's question-count range for the derived scale.
    # For extracted_word_count=100, your schema classifies as "small" and expects 4–6 questions.
    content = "\n".join(
        [
            "1. What is Q1?",
            "2. What is Q2?",
            "3. What is Q3?",
            "4. What is Q4?",
        ]
    )
    r = build_question_generation_inline_result(content=content, extracted_word_count=100, algorithm_version="v1")
    assert isinstance(r, QuestionGenerationInlineResult)
    assert r.scale.extracted_word_count == 100
    assert r.scale.classification is not None


def test_build_answer_generation_inline_result_includes_expected_question_count():
    r = build_answer_generation_inline_result(content="1. A", expected_question_count=1, algorithm_version="v1")
    assert isinstance(r, AnswerGenerationInlineResult)
    assert r.expected_question_count == 1


# -------------------------
# validate_analyzer_response
# -------------------------

def test_validate_analyzer_response_passes_when_request_matches_envelope_fields():
    req = AnalyzerRequest(
        action=FeatureType.summarize,
        input=_doc_input_valid(fmt=DocumentInputFormat.txt, text="Hello", extracted_word_count=1),
        payload=SummarizationRequest(feature=FeatureType.summarize),
        policy=_policy_for(FeatureType.summarize),
    )

    resp = AnalyzerResponse(
        action=FeatureType.summarize,
        input_format=DocumentInputFormat.txt,
        policy=req.policy,
        ui_language=req.ui_language,
        system_language=req.system_language,
        result=build_inline_txt_result(content="Short summary", algorithm_version="v1"),
    )

    out = validate_analyzer_response(resp, request=req)
    assert isinstance(out, AnalyzerResponse)
    assert out.action == FeatureType.summarize


def test_validate_analyzer_response_rejects_when_ui_language_does_not_match_request():
    req = AnalyzerRequest(
        action=FeatureType.summarize,
        input=_doc_input_valid(fmt=DocumentInputFormat.txt, text="Hello", extracted_word_count=1),
        payload=SummarizationRequest(feature=FeatureType.summarize),
        policy=_policy_for(FeatureType.summarize),
    )

    resp = AnalyzerResponse(
        action=FeatureType.summarize,
        input_format=DocumentInputFormat.txt,
        policy=req.policy,
        # Deliberate mismatch
        ui_language="french",
        system_language=req.system_language,
        result=build_inline_txt_result(content="Short summary", algorithm_version="v1"),
    )

    with pytest.raises(ValueError, match="Response ui_language must match request ui_language"):
        validate_analyzer_response(resp, request=req)


def test_validate_analyzer_response_rejects_when_action_mismatch():
    req = AnalyzerRequest(
        action=FeatureType.summarize,
        input=_doc_input_valid(fmt=DocumentInputFormat.txt, text="Hello", extracted_word_count=1),
        payload=SummarizationRequest(feature=FeatureType.summarize),
        policy=_policy_for(FeatureType.summarize),
    )

    resp = AnalyzerResponse(
        action=FeatureType.grammar_correct,  # mismatch
        input_format=DocumentInputFormat.txt,
        policy=req.policy,
        ui_language=req.ui_language,
        system_language=req.system_language,
        result=build_inline_txt_result(content="x", algorithm_version="v1"),
    )

    with pytest.raises(ValueError, match="Response action must match request action"):
        validate_analyzer_response(resp, request=req)
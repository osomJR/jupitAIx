from __future__ import annotations

from typing import Any, Mapping, Optional, Type, Union

from pydantic import BaseModel

from .schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    AnswerGenerationFileResult,
    AnswerGenerationInlineResult,
    AnswerGenerationRequest,
    ComplianceFileResult,
    ComplianceRequest,
    ComplianceReportVariant,
    ComplianceOutputFormat,
    ConversionRequest,
    DeterminismMetadata,
    DocumentFileOutputFormat,
    DocumentFileResult,
    DocumentInputFormat,
    DocumentPayload,
    DocumentSetPayload,
    ExplanationRequest,
    FeatureType,
    GrammarCorrectionRequest,
    InlineTextResult,
    MediaPayload,
    MediaType,
    OutputPolicy,
    QuestionGenerationFileResult,
    QuestionGenerationInlineResult,
    QuestionGenerationRequest,
    QuestionScale,
    QuestionScaleMetadata,
    RedactionRequest,
    DataMaskingRequest,
    StructuredDataOutputFormat,
    StructuredExtractionFileResult,
    StructuredExtractionRequest,
    StructuredExtractionResultShape,
    SummarizationRequest,
    TranscriptionRequest,
    TranslationRequest,
    TranscriptionResult,
    classify_word_count,
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
    FeatureType.structured_extract: StructuredExtractionRequest,
    FeatureType.compliance: ComplianceRequest,
    FeatureType.generate_questions: QuestionGenerationRequest,
    FeatureType.generate_answers: AnswerGenerationRequest,
}


def _iter_documents(input_artifact: Union[DocumentPayload, DocumentSetPayload, MediaPayload]) -> list[DocumentPayload]:
    if isinstance(input_artifact, DocumentPayload):
        return [input_artifact]
    if isinstance(input_artifact, DocumentSetPayload):
        return list(input_artifact.documents)
    return []


TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}


def validate_action_payload_consistency(request: AnalyzerRequest) -> None:
    """
    Explicit action-to-payload type guard that mirrors schema.py.
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
    for document in _iter_documents(request.input):
        if document.metadata.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection.")

    if isinstance(request.input, MediaPayload) and request.input.detected_language is not None:
        raise ValueError("detected_language must not be provided by client; server supplies detection.")


def validate_word_count_contract_when_present(request: AnalyzerRequest) -> None:
    """
    Keeps validation.py aligned with schema.py's narrowed word-count contract:
    extracted_word_count may be present on document metadata generally, but the
    1..1000 enforced range applies only to text-based AI document actions that
    require extracted text and word count.
    """
    if request.action not in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
        return

    if not isinstance(request.input, DocumentPayload):
        raise ValueError(f"{request.action.value} requires DocumentPayload input.")

    wc = request.input.metadata.extracted_word_count
    if wc is None:
        raise ValueError(f"{request.action.value} requires extracted_word_count.")
    if wc < 1:
        raise ValueError("extracted_word_count must be >= 1 for text-based AI processing actions.")

    # schema.classify_word_count is the authoritative contract check for the
    # supported word-count range used by question-scaling logic.
    classify_word_count(wc)


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


def validate_generate_questions_request(request: AnalyzerRequest) -> None:
    if request.action != FeatureType.generate_questions:
        return

    if not isinstance(request.input, DocumentPayload):
        raise ValueError("generate_questions requires DocumentPayload input.")

    wc = request.input.metadata.extracted_word_count
    if wc is None:
        raise ValueError("generate_questions requires extracted_word_count.")

    classify_word_count(wc)


def validate_generate_answers_request(request: AnalyzerRequest) -> None:
    """
    Supplemental checks for answer generation that remain consistent with schema.py:
    - DocumentPayload input only
    - AnswerGenerationRequest payload only
    - extracted_word_count must map to a valid question-scaling rule
    - supplied questions count must fit the schema scaling range
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
) -> AnalyzerRequest:
    """
    Full deterministic validation pipeline for incoming requests.

    Primary contract enforcement lives in schema.py via Pydantic validators.
    This function adds explicit alignment checks without inventing transport-level
    fields or workflow metadata that are not modeled in schema.py.
    """
    req = request if isinstance(request, AnalyzerRequest) else AnalyzerRequest.model_validate(request)

    validate_action_payload_consistency(req)
    validate_no_client_detected_language(req)
    validate_word_count_contract_when_present(req)
    validate_generate_questions_request(req)
    validate_generate_answers_request(req)

    if not isinstance(req.policy, OutputPolicy):
        raise ValueError("AnalyzerRequest.policy must be OutputPolicy.")

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

def build_transcription_result(
    *,
    content: str,
    pdf_artifact: DocumentFileResult,
    algorithm_version: Optional[str] = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        content=content,
        pdf_artifact=pdf_artifact,
        meta=_meta(algorithm_version=algorithm_version),
    )


def build_document_file_result(
    *,
    filename: str,
    output_format: DocumentFileOutputFormat,
    file_size_mb: float,
    storage_key: Optional[str] = None,
    download_url: Optional[str] = None,
    algorithm_version: Optional[str] = None,
) -> DocumentFileResult:
    return DocumentFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        storage_key=storage_key,
        download_url=download_url,
        meta=_meta(algorithm_version=algorithm_version),
    )


def build_structured_extraction_file_result(
    *,
    filename: str,
    output_format: StructuredDataOutputFormat,
    file_size_mb: float,
    result_shape: StructuredExtractionResultShape,
    selected_fields: Optional[list[str]] = None,
    storage_key: Optional[str] = None,
    download_url: Optional[str] = None,
    algorithm_version: Optional[str] = None,
) -> StructuredExtractionFileResult:
    return StructuredExtractionFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        result_shape=result_shape,
        selected_fields=selected_fields or [],
        storage_key=storage_key,
        download_url=download_url,
        meta=_meta(algorithm_version=algorithm_version),
    )


def build_compliance_file_result(
    *,
    filename: str,
    output_format: ComplianceOutputFormat,
    file_size_mb: float,
    report_variant: ComplianceReportVariant,
    storage_key: Optional[str] = None,
    download_url: Optional[str] = None,
    algorithm_version: Optional[str] = None,
) -> ComplianceFileResult:
    return ComplianceFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        report_variant=report_variant,
        storage_key=storage_key,
        download_url=download_url,
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
    output_format: DocumentFileOutputFormat,
    file_size_mb: float,
    extracted_word_count: int,
    storage_key: Optional[str] = None,
    download_url: Optional[str] = None,
    algorithm_version: Optional[str] = None,
) -> QuestionGenerationFileResult:
    classification = classify_word_count(extracted_word_count).classification
    return QuestionGenerationFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        storage_key=storage_key,
        download_url=download_url,
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
    output_format: DocumentFileOutputFormat,
    file_size_mb: float,
    expected_question_count: int,
    storage_key: Optional[str] = None,
    download_url: Optional[str] = None,
    algorithm_version: Optional[str] = None,
) -> AnswerGenerationFileResult:
    return AnswerGenerationFileResult(
        filename=filename,
        output_format=output_format,
        file_size_mb=file_size_mb,
        storage_key=storage_key,
        download_url=download_url,
        meta=_meta(algorithm_version=algorithm_version),
        expected_question_count=expected_question_count,
    )


# =========================
# Response validation
# =========================


def _expected_response_input_format(request: AnalyzerRequest):
    """
    AnalyzerResponse echoes input_format as:
      - DocumentInputFormat for single-document requests
      - "document_set" for DocumentSetPayload requests
      - "audio" or "video" for transcription requests
    """
    if isinstance(request.input, MediaPayload):
        return "audio" if request.input.media_type == MediaType.audio else "video"
    if isinstance(request.input, DocumentSetPayload):
        return "document_set"
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

        if request.action == FeatureType.structured_extract:
            assert isinstance(request.payload, StructuredExtractionRequest)
            if not isinstance(resp.result, StructuredExtractionFileResult):
                raise ValueError("structured_extract response must contain StructuredExtractionFileResult.")
            if resp.result.output_format != request.payload.output_format:
                raise ValueError("structured_extract response output_format must match request payload.output_format.")
            if resp.result.result_shape != request.payload.result_shape:
                raise ValueError("structured_extract response result_shape must match request payload.result_shape.")
            if resp.result.selected_fields != request.payload.selected_fields:
                raise ValueError("structured_extract response selected_fields must match request payload.selected_fields.")

        if request.action == FeatureType.compliance:
            assert isinstance(request.payload, ComplianceRequest)
            if not isinstance(resp.result, ComplianceFileResult):
                raise ValueError("compliance response must contain ComplianceFileResult.")
            if resp.result.report_variant != request.payload.report_variant:
                raise ValueError("compliance response report_variant must match request payload.report_variant.")
            expected_output_format = (
                ComplianceOutputFormat.json
                if request.payload.report_variant == ComplianceReportVariant.machine_readable_report
                else ComplianceOutputFormat.pdf
            )
            if resp.result.output_format != expected_output_format:
                raise ValueError("compliance response output_format is inconsistent with report_variant.")

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
    "validate_action_payload_consistency",
    "validate_no_client_detected_language",
    "validate_word_count_contract_when_present",
    "validate_question_scale",
    "get_question_range",
    "validate_generate_questions_request",
    "validate_generate_answers_request",
    "validate_analyzer_request",
    "build_inline_txt_result",
    "build_document_file_result",
    "build_structured_extraction_file_result",
    "build_compliance_file_result",
    "build_question_generation_inline_result",
    "build_question_generation_file_result",
    "build_answer_generation_inline_result",
    "build_answer_generation_file_result",
    "validate_analyzer_response",
]

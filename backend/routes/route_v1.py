from __future__ import annotations

import os
from typing import Any, Mapping, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from backend.errors import to_http_exception
from backend.auth0_dependencies import AuthenticatedUser, get_current_user
from backend.rate_limiter.dependencies import rate_limit_for_feature
from backend.upload import (
    UploadError,
    build_uploaded_document_payload,
    build_uploaded_media_payload,
)
from src.analyzer import Analyzer
from src.extraction import build_inline_text_payload
from src.processing.data_protection.orchestration import (
    ProtectedArtifactResult,
    process_privacy_action_and_persist,
)
from src.schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    AnswerGenerationRequest,
    ConversionOutputFormat,
    ConversionRequest,
    DataMaskingRequest,
    ExplanationRequest,
    FeatureType,
    GrammarCorrectionRequest,
    MediaType,
    OutputPolicy,
    QuestionGenerationRequest,
    RedactionMaskingDocumentType,
    RedactionRequest,
    SensitiveDataType,
    SummarizationRequest,
    SystemLanguage,
    TranscriptionRequest,
    TranslationRequest,
)

API_V1_ANALYZER_PREFIX = "/analyzer"

router = APIRouter(prefix=API_V1_ANALYZER_PREFIX, tags=["analyzer-v1"])
analyzer = Analyzer()

TRANSFORMED_ACTIONS = {
    FeatureType.convert,
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.transcribe,
    FeatureType.redact,
    FeatureType.data_mask,
}

GENERATED_ACTIONS = {
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

DEFAULT_PRIVACY_OUTPUT_DIR = os.getenv("PRIVACY_OUTPUT_DIR", "outputs/privacy")
DEFAULT_GOOGLE_SDP_LOCATION = os.getenv("GOOGLE_SDP_LOCATION", "global")


def _policy_for_action(action: FeatureType) -> OutputPolicy:
    if action in TRANSFORMED_ACTIONS:
        return OutputPolicy(structure_preservation=True)
    if action in GENERATED_ACTIONS:
        return OutputPolicy(structure_preservation=False)
    raise ValueError(f"Unsupported action: {action}")


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "invalid_request",
            "message": message,
        },
    )


def _service_unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "error": "service_unavailable",
            "message": message,
        },
    )


def _build_document_input(
    *,
    action: FeatureType,
    file: UploadFile | None,
    text: str | None,
):
    has_file = file is not None
    has_text = text is not None and text.strip() != ""

    if has_file and has_text:
        raise _bad_request("Provide either file or text, not both.")
    if not has_file and not has_text:
        raise _bad_request("Either file or text is required.")

    try:
        if has_file:
            return build_uploaded_document_payload(action=action, upload=file)  # type: ignore[arg-type]
        return build_inline_text_payload(text=text.strip())  # type: ignore[union-attr]
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc


def _run_request(request: Union[AnalyzerRequest, Mapping[str, Any]]) -> AnalyzerResponse:
    try:
        return analyzer.analyze(request)
    except HTTPException:
        raise
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc
    except TypeError as exc:
        raise _bad_request(str(exc)) from exc


def _google_sdp_project_id() -> str:
    project_id = os.getenv("GOOGLE_SDP_PROJECT_ID", "").strip()
    if not project_id:
        raise _service_unavailable(
            "Google Sensitive Data Protection is not configured. "
            "Set GOOGLE_SDP_PROJECT_ID for redaction and data masking."
        )
    return project_id


def _privacy_source_path(input_payload) -> str:
    filename = getattr(input_payload, "filename", None)
    if not isinstance(filename, str) or not filename.strip():
        raise _bad_request("Uploaded privacy document is missing its persisted file path.")
    return filename.strip()


def _run_privacy_request(
    request: AnalyzerRequest,
    *,
    source_path: str,
) -> ProtectedArtifactResult:
    try:
        return process_privacy_action_and_persist(
            request,
            source_path=source_path,
            output_dir=DEFAULT_PRIVACY_OUTPUT_DIR,
            project_id=_google_sdp_project_id(),
            location=DEFAULT_GOOGLE_SDP_LOCATION,
        )
    except HTTPException as exc:
        raise to_http_exception(exc) from exc
    


@router.post(
    "/convert",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.convert))],
)
def convert_route(
    file: UploadFile = File(...),
    output_format: ConversionOutputFormat = Form(...),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    try:
        input_payload = build_uploaded_document_payload(
            action=FeatureType.convert,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    request = AnalyzerRequest(
        action=FeatureType.convert,
        input=input_payload,
        payload=ConversionRequest(
            feature=FeatureType.convert,
            output_format=output_format,
        ),
        policy=_policy_for_action(FeatureType.convert),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/summarize",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.summarize))],
)
def summarize_route(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.summarize,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.summarize,
        input=input_payload,
        payload=SummarizationRequest(feature=FeatureType.summarize),
        policy=_policy_for_action(FeatureType.summarize),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/grammar-correct",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.grammar_correct))],
)
def grammar_correct_route(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.grammar_correct,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.grammar_correct,
        input=input_payload,
        payload=GrammarCorrectionRequest(feature=FeatureType.grammar_correct),
        policy=_policy_for_action(FeatureType.grammar_correct),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/translate",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.translate))],
)
def translate_route(
    target_language: str = Form(...),
    source_language: str = Form("auto"),
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.translate,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.translate,
        input=input_payload,
        payload=TranslationRequest(
            feature=FeatureType.translate,
            source_language=source_language,
            target_language=target_language,
        ),
        policy=_policy_for_action(FeatureType.translate),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/transcribe",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.transcribe))],
)
def transcribe_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    media_type: MediaType = Form(...),
    duration_seconds: int = Form(...),
    preserve_filler_words: bool = Form(True),
    remove_background_noise: bool = Form(False),
    diarize_speakers: bool = Form(True),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    try:
        input_payload = build_uploaded_media_payload(
            upload=file,
            media_type=media_type,
            duration_seconds=duration_seconds,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    request = AnalyzerRequest(
        action=FeatureType.transcribe,
        input=input_payload,
        payload=TranscriptionRequest(
            feature=FeatureType.transcribe,
            preserve_filler_words=preserve_filler_words,
            remove_background_noise=remove_background_noise,
            diarize_speakers=diarize_speakers,
        ),
        policy=_policy_for_action(FeatureType.transcribe),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/explain",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.explain))],
)
def explain_route(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    allow_external_knowledge: bool = Form(False),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.explain,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.explain,
        input=input_payload,
        payload=ExplanationRequest(
            feature=FeatureType.explain,
            allow_external_knowledge=allow_external_knowledge,
        ),
        policy=_policy_for_action(FeatureType.explain),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/generate-questions",
    response_model=AnalyzerResponse,
    # dependencies=[Depends(rate_limit_for_feature(FeatureType.generate_questions))],
)
def generate_questions_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.generate_questions,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.generate_questions,
        input=input_payload,
        payload=QuestionGenerationRequest(
            feature=FeatureType.generate_questions,
        ),
        policy=_policy_for_action(FeatureType.generate_questions),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/generate-answers",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.generate_answers))],
)
def generate_answers_route(
    questions: list[str] = Form(...),
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
    generate_questions_token: str | None = Form(default=None),
    generate_questions_completed: bool = Form(False),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.generate_answers,
        file=file,
        text=text,
    )

    request_data: dict[str, Any] = {
        "action": FeatureType.generate_answers,
        "input": input_payload.model_dump(mode="python"),
        "payload": AnswerGenerationRequest(
            feature=FeatureType.generate_answers,
            questions=questions,
        ).model_dump(mode="python"),
        "policy": _policy_for_action(FeatureType.generate_answers).model_dump(mode="python"),
        "system_language": system_language,
    }

    if generate_questions_token is not None and generate_questions_token.strip():
        request_data["generate_questions_token"] = generate_questions_token.strip()

    if generate_questions_completed:
        request_data["workflow"] = {"generate_questions_completed": True}

    return _run_request(request_data)


@router.post(
    "/redact",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.redact))],
)
def redact_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    document_type: RedactionMaskingDocumentType | None = Form(default=None),
    target_data: list[SensitiveDataType] | None = Form(default=None),
    review_exclusions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> ProtectedArtifactResult:
    try:
        input_payload = build_uploaded_document_payload(
            action=FeatureType.redact,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    payload_kwargs: dict[str, Any] = {"feature": FeatureType.redact}
    if document_type is not None:
        payload_kwargs["document_type"] = document_type
    if target_data:
        payload_kwargs["target_data"] = target_data
    if review_exclusions:
        payload_kwargs["review_exclusions"] = [item.strip() for item in review_exclusions if item and item.strip()]

    request = AnalyzerRequest(
        action=FeatureType.redact,
        input=input_payload,
        payload=RedactionRequest(**payload_kwargs),
        policy=_policy_for_action(FeatureType.redact),
        system_language=system_language,
    )
    return _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
    )


@router.post(
    "/data-mask",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.data_mask))],
)
def data_mask_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    document_type: RedactionMaskingDocumentType | None = Form(default=None),
    target_data: list[SensitiveDataType] | None = Form(default=None),
    review_exclusions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> ProtectedArtifactResult:
    try:
        input_payload = build_uploaded_document_payload(
            action=FeatureType.data_mask,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    payload_kwargs: dict[str, Any] = {"feature": FeatureType.data_mask}
    if document_type is not None:
        payload_kwargs["document_type"] = document_type
    if target_data:
        payload_kwargs["target_data"] = target_data
    if review_exclusions:
        payload_kwargs["review_exclusions"] = [item.strip() for item in review_exclusions if item and item.strip()]

    request = AnalyzerRequest(
        action=FeatureType.data_mask,
        input=input_payload,
        payload=DataMaskingRequest(**payload_kwargs),
        policy=_policy_for_action(FeatureType.data_mask),
        system_language=system_language,
    )
    return _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
    )


__all__ = ["router", "API_V1_ANALYZER_PREFIX"]

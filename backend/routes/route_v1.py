from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Union
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.auth0_dependencies import AuthenticatedUser, get_current_user
from backend.errors import to_http_exception
from backend.rate_limiter.dependencies import rate_limit_for_feature
from backend.upload import (
    UploadError,
    build_uploaded_document_payload,
    build_uploaded_media_payload,
)
from src.analyzer import Analyzer
from src.extraction import build_inline_text_payload
from src.processing.conversion.convert import convert_document
from src.processing.data_protection.data_masking.data_mask import (
    preview_data_mask_candidates,
)
from src.processing.data_protection.orchestration import (
    ProtectedArtifactResult,
    process_privacy_action_and_persist,
)
from src.processing.data_protection.redaction.redact import (
    preview_redaction_candidates,
)

from src.processing.compliance.compliance import run_compliance, preview_compliance
from src.processing.compliance.registry import RuleRegistryError
from src.processing.structured_extraction.structured_extraction import (
    run_structured_extraction, run_structured_extraction_with_preview,
)

from src.schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    AnswerGenerationRequest,
    ComplianceJurisdiction,
    ComplianceRegulatoryDomain,
    ComplianceReportVariant,
    ComplianceRequest,
    ComplianceSectorPack,
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
    StructuredDataOutputFormat,
    StructuredExtractionDocumentClass,
    StructuredExtractionRequest,
    StructuredExtractionResultShape,
    SummarizationRequest,
    SystemLanguage,
    TranscriptionRequest,
    TranslationRequest,
)

from src.storage.artifacts import LocalArtifactStorage, guess_content_type

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
    FeatureType.structured_extract,
    FeatureType.compliance,
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
    custom_redactions: list[str] | None = None,
) -> ProtectedArtifactResult:
    try:
        return process_privacy_action_and_persist(
            request,
            source_path=source_path,
            output_dir=DEFAULT_PRIVACY_OUTPUT_DIR,
            project_id=_google_sdp_project_id(),
            location=DEFAULT_GOOGLE_SDP_LOCATION,
            custom_redactions=custom_redactions,
        )
    except HTTPException as exc:
        raise to_http_exception(exc) from exc
    
def _download_url_for_storage_key(storage_key: str | None) -> str | None:
    if not isinstance(storage_key, str) or not storage_key.strip():
        return None

    key = storage_key.strip().replace("\\", "/")

    # Remove prefixes if the caller accidentally stored a full/partial artifact path.
    key = key.removeprefix("/api/analyzer/artifacts/")
    key = key.removeprefix("/api/v1/analyzer/artifacts/")
    key = key.removeprefix("/artifacts/")
    key = key.removeprefix("artifacts/")

    return f"/api/analyzer/artifacts/{key}"


def _ensure_download_url(response: AnalyzerResponse) -> AnalyzerResponse:
    result = response.result
    storage_key = getattr(result, "storage_key", None)
    download_url = getattr(result, "download_url", None)

    if storage_key and not download_url and hasattr(result, "download_url"):
        result.download_url = _download_url_for_storage_key(storage_key)

    return response

def _run_structured_extraction_request_with_preview(
    request: AnalyzerRequest,
) -> dict[str, Any]:
    try:
        execution = run_structured_extraction_with_preview(request)
        response = _ensure_download_url(execution.response)

        preview_rows_limit = 50
        preview_rows = execution.preview_rows[:preview_rows_limit]

        return {
            "analyzer_response": response.model_dump(mode="json"),
            "preview_payload": execution.preview_payload,
            "preview_rows": preview_rows,
            "preview_truncated": len(execution.preview_rows) > preview_rows_limit,
        }

    except HTTPException:
        raise
    except RuleRegistryError as exc:
        raise _bad_request(str(exc)) from exc
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc
    except TypeError as exc:
        raise _bad_request(str(exc)) from exc
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc

def _run_standalone_feature_request(
    request: AnalyzerRequest,
) -> AnalyzerResponse:
    try:
        if request.action == FeatureType.structured_extract:
            return _ensure_download_url(run_structured_extraction(request))

        if request.action == FeatureType.compliance:
            return _ensure_download_url(run_compliance(request))

        raise ValueError(f"Unsupported standalone feature: {request.action.value}")
    except HTTPException:
        raise
    except RuleRegistryError as exc:
        raise _bad_request(str(exc)) from exc
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc
    except TypeError as exc:
        raise _bad_request(str(exc)) from exc
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc


def _clean_repeated_strings(values: list[str] | None) -> list[str]:
    if not values:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)

    return cleaned

def _normalize_compliance_sector_packs(
    sector_packs: list[ComplianceSectorPack] | None,
) -> list[ComplianceSectorPack]:
    core_pack = ComplianceSectorPack.core_control_library
    legacy_core_pack = getattr(
        ComplianceSectorPack,
        "nigeria_core_control_library",
        None,
    )

    resolved: list[ComplianceSectorPack] = []
    seen: set[ComplianceSectorPack] = set()

    for pack in list(sector_packs or []):
        if legacy_core_pack is not None and pack == legacy_core_pack:
            pack = core_pack

        if pack not in seen:
            resolved.append(pack)
            seen.add(pack)

    if core_pack not in seen:
        resolved.insert(0, core_pack)

    return resolved

def _build_compliance_request(
    *,
    file: UploadFile,
    jurisdiction: ComplianceJurisdiction,
    sector_packs: list[ComplianceSectorPack] | None,
    regulatory_domains: list[ComplianceRegulatoryDomain] | None,
    report_variant: ComplianceReportVariant,
    system_language: SystemLanguage,
) -> AnalyzerRequest:
    try:
        input_payload = build_uploaded_document_payload(
            action=FeatureType.compliance,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc

    resolved_sector_packs = _normalize_compliance_sector_packs(
        sector_packs or _default_compliance_sector_packs(jurisdiction)
    )

    payload = ComplianceRequest(
        feature=FeatureType.compliance,
        jurisdiction=jurisdiction,
        sector_packs=resolved_sector_packs,
        regulatory_domains=regulatory_domains or [],
        report_variant=report_variant,
        require_human_review=True,
    )

    return AnalyzerRequest(
        action=FeatureType.compliance,
        input=input_payload,
        payload=payload,
        policy=_policy_for_action(FeatureType.compliance),
        system_language=system_language,
    )

def _default_compliance_sector_packs(
    jurisdiction: ComplianceJurisdiction,
) -> list[ComplianceSectorPack]:
    return [ComplianceSectorPack.core_control_library]

def _privacy_payload_kwargs(
    *,
    feature: FeatureType,
    document_type: RedactionMaskingDocumentType | None,
    target_data: list[SensitiveDataType] | None,
    review_exclusions: list[str] | None,
) -> dict[str, Any]:
    payload_kwargs: dict[str, Any] = {"feature": feature}
    if document_type is not None:
        payload_kwargs["document_type"] = document_type
    if target_data:
        payload_kwargs["target_data"] = target_data
    if review_exclusions:
        payload_kwargs["review_exclusions"] = [
            item.strip() for item in review_exclusions if item and item.strip()
        ]
    return payload_kwargs


def _build_privacy_request(
    *,
    action: FeatureType,
    file: UploadFile,
    document_type: RedactionMaskingDocumentType | None,
    target_data: list[SensitiveDataType] | None,
    review_exclusions: list[str] | None,
    system_language: SystemLanguage,
) -> tuple[Any, AnalyzerRequest]:
    try:
        input_payload = build_uploaded_document_payload(
            action=action,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    payload_kwargs = _privacy_payload_kwargs(
        feature=action,
        document_type=document_type,
        target_data=target_data,
        review_exclusions=review_exclusions,
    )

    payload = (
        RedactionRequest(**payload_kwargs)
        if action == FeatureType.redact
        else DataMaskingRequest(**payload_kwargs)
    )

    request = AnalyzerRequest(
        action=action,
        input=input_payload,
        payload=payload,
        policy=_policy_for_action(action),
        system_language=system_language,
    )
    return input_payload, request


def _serialize_candidates(candidates: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        serialized.append(
            {
                "id": f"{candidate.label}|{candidate.source}|{candidate.quote}|{index}",
                "label": candidate.label,
                "quote": candidate.quote,
                "occurrences": candidate.occurrences,
                "source": candidate.source,
            }
        )
    return serialized


def _build_docx_preview_artifact(processed: ProtectedArtifactResult) -> dict[str, Any] | None:
    original_name = processed.artifact.original_artifact_name.lower()
    if not original_name.endswith(".docx"):
        return None

    preview = convert_document(
        input_format="docx",
        output_format="pdf",
        source_reference=processed.artifact.stored_path,
        source_name_hint=processed.artifact.original_artifact_name,
    )

    preview_storage_key = preview.storage_key
    preview_download_url = preview.download_url
    if not preview_download_url and preview_storage_key:
        preview_download_url = f"/api/v1/analyzer/artifacts/{preview_storage_key}"

    return {
        "filename": preview.file_name,
        "storage_key": preview_storage_key,
        "download_url": preview_download_url,
        "content_type": "application/pdf",
    }


def _serialize_processed_result(processed: ProtectedArtifactResult) -> dict[str, Any]:
    return {
        "analyzer_response": processed.analyzer_response.model_dump(mode="python"),
        "artifact": asdict(processed.artifact),
        "generated_output_path": processed.generated_output_path,
        "preview_artifact": _build_docx_preview_artifact(processed),
    }


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
    dependencies=[Depends(rate_limit_for_feature(FeatureType.generate_questions))],
)
def generate_questions_route(
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
        payload=QuestionGenerationRequest(feature=FeatureType.generate_questions),
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
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.generate_answers,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.generate_answers,
        input=input_payload,
        payload=AnswerGenerationRequest(feature=FeatureType.generate_answers),
        policy=_policy_for_action(FeatureType.generate_answers),
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/redact/review",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.redact))],
)
def redact_review_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    document_type: RedactionMaskingDocumentType | None = Form(default=None),
    target_data: list[SensitiveDataType] | None = Form(default=None),
    review_exclusions: list[str] | None = Form(default=None),
    custom_redactions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    input_payload, request = _build_privacy_request(
        action=FeatureType.redact,
        file=file,
        document_type=document_type,
        target_data=target_data,
        review_exclusions=review_exclusions,
        system_language=system_language,
    )

    cleaned_custom_redactions = _clean_repeated_strings(custom_redactions)

    processed = _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
        custom_redactions=cleaned_custom_redactions,
    )

    candidates = preview_redaction_candidates(
        request,
        project_id=_google_sdp_project_id(),
        location=DEFAULT_GOOGLE_SDP_LOCATION,
        custom_redactions=cleaned_custom_redactions,
    )

    return {
        **_serialize_processed_result(processed),
        "candidates": _serialize_candidates(candidates),
    }


@router.post(
    "/data-mask/review",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.data_mask))],
)
def data_mask_review_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    document_type: RedactionMaskingDocumentType | None = Form(default=None),
    target_data: list[SensitiveDataType] | None = Form(default=None),
    review_exclusions: list[str] | None = Form(default=None),
    custom_redactions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    input_payload, request = _build_privacy_request(
        action=FeatureType.data_mask,
        file=file,
        document_type=document_type,
        target_data=target_data,
        review_exclusions=review_exclusions,
        system_language=system_language,
    )

    cleaned_custom_redactions = _clean_repeated_strings(custom_redactions)

    processed = _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
        custom_redactions=cleaned_custom_redactions,
    )

    candidates = preview_data_mask_candidates(
        request,
        project_id=_google_sdp_project_id(),
        location=DEFAULT_GOOGLE_SDP_LOCATION,
        custom_redactions=cleaned_custom_redactions,
    )

    return {
        **_serialize_processed_result(processed),
        "candidates": _serialize_candidates(candidates),
    }


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
    custom_redactions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    input_payload, request = _build_privacy_request(
        action=FeatureType.redact,
        file=file,
        document_type=document_type,
        target_data=target_data,
        review_exclusions=review_exclusions,
        system_language=system_language,
    )

    processed = _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
        custom_redactions=_clean_repeated_strings(custom_redactions),
    )
    return _serialize_processed_result(processed)


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
    custom_redactions: list[str] | None = Form(default=None),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    input_payload, request = _build_privacy_request(
        action=FeatureType.data_mask,
        file=file,
        document_type=document_type,
        target_data=target_data,
        review_exclusions=review_exclusions,
        system_language=system_language,
    )

    processed = _run_privacy_request(
        request,
        source_path=_privacy_source_path(input_payload),
        custom_redactions=_clean_repeated_strings(custom_redactions),
    )
    return _serialize_processed_result(processed)

@router.post(
    "/structured-extraction",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.structured_extract))],
)
def structured_extraction_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    document_classes: list[StructuredExtractionDocumentClass] | None = Form(default=None),
    selected_fields: list[str] | None = Form(default=None),
    output_format: StructuredDataOutputFormat = Form(StructuredDataOutputFormat.json),
    result_shape: StructuredExtractionResultShape = Form(
        StructuredExtractionResultShape.machine_readable
    ),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    try:
        input_payload = build_uploaded_document_payload(
            action=FeatureType.structured_extract,
            upload=file,
        )
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    payload = StructuredExtractionRequest(
        feature=FeatureType.structured_extract,
        document_classes=document_classes or [],
        selected_fields=_clean_repeated_strings(selected_fields),
        output_format=output_format,
        result_shape=result_shape,
        allow_external_knowledge=False,
        require_human_review=True,
    )

    request = AnalyzerRequest(
        action=FeatureType.structured_extract,
        input=input_payload,
        payload=payload,
        policy=_policy_for_action(FeatureType.structured_extract),
        system_language=system_language,
    )

    return _run_structured_extraction_request_with_preview(request)

@router.post(
    "/compliance",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.compliance))],
)
def compliance_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    jurisdiction: ComplianceJurisdiction = Form(ComplianceJurisdiction.nigeria),
    sector_packs: list[ComplianceSectorPack] | None = Form(default=None),
    regulatory_domains: list[ComplianceRegulatoryDomain] | None = Form(default=None),
    report_variant: ComplianceReportVariant = Form(
        ComplianceReportVariant.human_readable_report
    ),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> AnalyzerResponse:
    request = _build_compliance_request(
        file=file,
        jurisdiction=jurisdiction,
        sector_packs=sector_packs,
        regulatory_domains=regulatory_domains,
        report_variant=report_variant,
        system_language=system_language,
    )

    return _run_standalone_feature_request(request)

@router.post(
    "/compliance/preview",
    dependencies=[Depends(rate_limit_for_feature(FeatureType.compliance))],
)
def compliance_preview_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
    file: UploadFile = File(...),
    jurisdiction: ComplianceJurisdiction = Form(ComplianceJurisdiction.nigeria),
    sector_packs: list[ComplianceSectorPack] | None = Form(default=None),
    regulatory_domains: list[ComplianceRegulatoryDomain] | None = Form(default=None),
    report_variant: ComplianceReportVariant = Form(
        ComplianceReportVariant.human_readable_report
    ),
    system_language: SystemLanguage = Form(SystemLanguage.english),
) -> dict[str, Any]:
    try:
        request = _build_compliance_request(
            file=file,
            jurisdiction=jurisdiction,
            sector_packs=sector_packs,
            regulatory_domains=regulatory_domains,
            report_variant=report_variant,
            system_language=system_language,
        )

        preview = preview_compliance(request)
        report = preview.report.model_dump(mode="json")

        return {
            "preview_markdown": preview.preview_markdown,
            "report": report,
            "counts": report.get("counts"),
            "rule_results": report.get("rule_results", []),
            "human_review": preview.human_review.model_dump(mode="json"),
        }

    except HTTPException:
        raise
    except RuleRegistryError as exc:
        raise _bad_request(str(exc)) from exc
    except UploadError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    except FileNotFoundError as exc:
        raise _bad_request(str(exc)) from exc
    except TypeError as exc:
        raise _bad_request(str(exc)) from exc
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc

@router.api_route("/artifacts/{storage_key:path}", methods=["GET", "HEAD"])
def download_artifact(
    storage_key: str,
    disposition: Literal["attachment", "inline"] = "attachment",
):
    storage = LocalArtifactStorage()

    content_disposition_type = (
        "inline" if disposition == "inline" else "attachment"
    )

    def _file_response(path: Path):
        response = FileResponse(
            path=str(path),
            media_type=guess_content_type(str(path)),
        )

        filename = path.name.replace('"', "")
        encoded_filename = quote(filename)

        response.headers["Content-Disposition"] = (
            f'{content_disposition_type}; filename="{filename}"; '
            f"filename*=UTF-8''{encoded_filename}"
        )

        return response

    try:
        path = storage.resolve_storage_key(storage_key)
        if path.exists() and path.is_file():
            return _file_response(path)
    except ValueError:
        pass

    normalized_key = storage_key.strip().replace("\\", "/")
    candidate = Path(normalized_key)

    if candidate.is_absolute():
        raise HTTPException(status_code=400, detail="Artifact path must be relative.")

    if any(part == ".." for part in candidate.parts):
        raise HTTPException(
            status_code=400,
            detail="Artifact path must not contain parent-directory traversal.",
        )

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    allowed_roots = {
        Path("artifacts").resolve(),
        Path("outputs").resolve(),
    }

    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(
            status_code=400,
            detail="Artifact path is outside the allowed artifact directories.",
        )

    return _file_response(resolved)

__all__ = ["router", "API_V1_ANALYZER_PREFIX"]
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.analyzer import Analyzer
from src.extraction import build_inline_text_payload
from src.schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    AnswerGenerationRequest,
    ConversionOutputFormat,
    ConversionRequest,
    ExplanationRequest,
    FeatureType,
    GrammarCorrectionRequest,
    MediaType,
    OutputPolicy,
    QuestionGenerationRequest,
    SummarizationRequest,
    SystemLanguageMode,
    TranscriptionRequest,
    TranslationRequest,
    UILanguage,
)
from backend.upload import (
    UploadError,
    build_uploaded_document_payload,
    build_uploaded_media_payload,
)
from backend.rate_limiter.dependencies import rate_limit_for_feature


API_V1_ANALYZER_PREFIX = "/analyzer"

router = APIRouter(prefix=API_V1_ANALYZER_PREFIX, tags=["analyzer-v1"])
analyzer = Analyzer()


TRANSFORMED_ACTIONS = {
    FeatureType.convert,
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.transcribe,
}

GENERATED_ACTIONS = {
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}


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


def _run_request(request: AnalyzerRequest) -> AnalyzerResponse:
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


@router.post(
    "/convert",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.convert))],
)
def convert_route(
    file: UploadFile = File(...),
    output_format: ConversionOutputFormat = Form(...),
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
        system_language=system_language,
    )
    return _run_request(request)


@router.post(
    "/transcribe",
    response_model=AnalyzerResponse,
    dependencies=[Depends(rate_limit_for_feature(FeatureType.transcribe))],
)
def transcribe_route(
    file: UploadFile = File(...),
    media_type: MediaType = Form(...),
    duration_seconds: int = Form(...),
    preserve_filler_words: bool = Form(True),
    remove_background_noise: bool = Form(False),
    diarize_speakers: bool = Form(True),
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
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
        ui_language=ui_language,
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
    ui_language: UILanguage = Form(UILanguage.english),
    system_language: SystemLanguageMode = Form(SystemLanguageMode.auto),
) -> AnalyzerResponse:
    input_payload = _build_document_input(
        action=FeatureType.generate_answers,
        file=file,
        text=text,
    )
    request = AnalyzerRequest(
        action=FeatureType.generate_answers,
        input=input_payload,
        payload=AnswerGenerationRequest(
            feature=FeatureType.generate_answers,
            questions=questions,
        ),
        policy=_policy_for_action(FeatureType.generate_answers),
        ui_language=ui_language,
        system_language=system_language,
    )
    return _run_request(request)


__all__ = ["router", "API_V1_ANALYZER_PREFIX"]
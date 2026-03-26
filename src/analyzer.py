from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .processing.llm.summarize import summarize_text
from .processing.llm.grammar_correct import grammar_correct_text
from .processing.llm.translate import translate_text
from .processing.llm.explain import explain_text
from .processing.llm.generate_questions import generate_questions_text
from .processing.llm.generate_answers import generate_answers_text
from .processing.asr.transcribe import transcribe_media
from .processing.conversion.convert import convert_document
from .processing.llm.writer import write_document
from .schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    FeatureType,
    SystemLanguage,
    DocumentPayload,
    MediaPayload,
    DocumentInputFormat,
    MediaType,
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    TranscriptionRequest,
    ExplanationRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
    InlineTextResult,
    FileResult,
    FileOutputFormat,
    QuestionGenerationInlineResult,
    QuestionGenerationFileResult,
    AnswerGenerationInlineResult,
    AnswerGenerationFileResult,
)
from .validation import (
    validate_analyzer_request,
    validate_analyzer_response,
    build_file_result,
    build_question_generation_inline_result,
    build_question_generation_file_result,
    get_question_range,
    build_answer_generation_inline_result,
    build_answer_generation_file_result,
    build_inline_txt_result,
)


@dataclass(frozen=True)
class AnalyzerConfig:
    """
    Deterministic configuration knobs.

    - algorithm_version: echoed into DeterminismMetadata.meta.algorithm_version
    - include_language_fields: if True, analyzer sets detected_language/output_language where valid by contract.
    """
    algorithm_version: Optional[str] = None
    include_language_fields: bool = True


class Analyzer:
    """
    Contract-first analyzer:
      - validates requests using validation.validate_analyzer_request (schema is primary enforcer).
      - constructs AnalyzerResponse + Result objects that satisfy schema response validators.
      - optionally echoes detected_language/output_language within strict boundaries.

    Note: This module is intentionally deterministic and side-effect free.
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

    def analyze(
        self,
        request: Union[AnalyzerRequest, Mapping[str, Any]],
    ) -> AnalyzerResponse:
        req = validate_analyzer_request(request)

        if req.action == FeatureType.convert:
            result = self._handle_convert(req)
            resp = self._build_response(req, result=result, detected_language=None, output_language=None)

        elif req.action == FeatureType.transcribe:
            result = self._handle_transcribe(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.summarize:
            result = self._handle_summarize(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.grammar_correct:
            result = self._handle_grammar(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.translate:
            result = self._handle_translate(req)
            detected, out = self._language_fields_for_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.explain:
            result = self._handle_explain(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.generate_questions:
            result = self._handle_generate_questions(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        elif req.action == FeatureType.generate_answers:
            result = self._handle_generate_answers(req)
            detected, out = self._maybe_language_fields_for_non_translate(req)
            resp = self._build_response(req, result=result, detected_language=detected, output_language=out)

        else:
            raise ValueError(f"Unsupported action: {req.action}")

        resp = validate_analyzer_response(resp, request=req)
        return resp

    def _handle_convert(self, req: AnalyzerRequest) -> FileResult:
        if not isinstance(req.payload, ConversionRequest):
            raise ValueError("convert requires ConversionRequest payload.")
        if not isinstance(req.input, DocumentPayload):
            raise ValueError("convert requires DocumentPayload input.")
        source_reference = getattr(req.input, "filename", None)
        if not source_reference:
            raise ValueError("convert requires a real source file reference on DocumentPayload.filename.")

        source_name_hint = Path(source_reference).name
        artifact = convert_document(
            input_format=req.input.metadata.input_format.value,
            output_format=req.payload.output_format.value,
            source_reference=source_reference,
            source_name_hint=source_name_hint,
        )
        return build_file_result(
            filename=artifact.file_name,
            output_format=_conversion_to_file_output(artifact.file_extension),
            file_size_mb=artifact.file_size_mb,
            algorithm_version=self.config.algorithm_version,
        )

    def _handle_transcribe(self, req: AnalyzerRequest) -> InlineTextResult:
        if not isinstance(req.payload, TranscriptionRequest):
            raise ValueError("transcribe requires TranscriptionRequest payload.")
        if not isinstance(req.input, MediaPayload):
            raise ValueError("transcribe requires MediaPayload input.")

        content = transcribe_media(
            media_type="audio" if req.input.media_type == MediaType.audio else "video",
            media_format=req.input.media_format.value,
            file_reference=req.input.filename or (
                "audio-input" if req.input.media_type == MediaType.audio else "video-input"
            ),
            preserve_filler_words=req.payload.preserve_filler_words,
            remove_background_noise=req.payload.remove_background_noise,
            diarize_speakers=req.payload.diarize_speakers,
        )
        return build_inline_txt_result(
            content=content,
            algorithm_version=self.config.algorithm_version,
        )

    def _handle_summarize(self, req: AnalyzerRequest) -> Union[InlineTextResult, FileResult]:
        if not isinstance(req.payload, SummarizationRequest):
            raise ValueError("summarize requires SummarizationRequest payload.")
        return self._doc_action_inline_or_file(
            req,
            inline_builder=summarize_text,
            file_suffix="summary",
        )

    def _handle_grammar(self, req: AnalyzerRequest) -> Union[InlineTextResult, FileResult]:
        if not isinstance(req.payload, GrammarCorrectionRequest):
            raise ValueError("grammar_correct requires GrammarCorrectionRequest payload.")
        return self._doc_action_inline_or_file(
            req,
            inline_builder=grammar_correct_text,
            file_suffix="grammar",
        )

    def _handle_translate(self, req: AnalyzerRequest) -> Union[InlineTextResult, FileResult]:
        if not isinstance(req.payload, TranslationRequest):
            raise ValueError("translate requires TranslationRequest payload.")
        return self._doc_action_inline_or_file(
            req,
            inline_builder=lambda text: translate_text(
                text,
                source_language=req.payload.source_language,
                target_language=req.payload.target_language,
            ),
            file_suffix="translation",
        )

    def _handle_explain(self, req: AnalyzerRequest) -> Union[InlineTextResult, FileResult]:
        if not isinstance(req.payload, ExplanationRequest):
            raise ValueError("explain requires ExplanationRequest payload.")
        return self._doc_action_inline_or_file(
            req,
            inline_builder=explain_text,
            file_suffix="explanation",
        )

    def _handle_generate_questions(
        self, req: AnalyzerRequest
    ) -> Union[QuestionGenerationInlineResult, QuestionGenerationFileResult]:
        if not isinstance(req.payload, QuestionGenerationRequest):
            raise ValueError("generate_questions requires QuestionGenerationRequest payload.")
        if not isinstance(req.input, DocumentPayload):
            raise ValueError("generate_questions requires DocumentPayload input.")
        if req.input.metadata.extracted_word_count is None:
            raise ValueError("generate_questions requires extracted_word_count.")

        wc = int(req.input.metadata.extracted_word_count)
        q_min, q_max = get_question_range(wc)
        content = generate_questions_text(
            req.input.text,
            min_questions=q_min,
            max_questions=q_max,
        )

        in_fmt = req.input.metadata.input_format
        if in_fmt == DocumentInputFormat.txt:
            return build_question_generation_inline_result(
                content=content,
                extracted_word_count=wc,
                algorithm_version=self.config.algorithm_version,
            )

        if in_fmt in (DocumentInputFormat.pdf, DocumentInputFormat.docx):
            out_fmt = FileOutputFormat.pdf if in_fmt == DocumentInputFormat.pdf else FileOutputFormat.docx
            filename = f"{_safe_basename(in_fmt.value)}.questions.{out_fmt.value}"

            written = write_document(
                content=content,
                output_format=out_fmt.value,
                output_name=filename,
            )
            return build_question_generation_file_result(
                filename=written.file_name,
                output_format=out_fmt,
                file_size_mb=written.file_size_mb,
                extracted_word_count=wc,
                algorithm_version=self.config.algorithm_version,
            )

        raise ValueError("generate_questions only supports input formats: pdf, docx, txt (strict v1 contract).")

    def _handle_generate_answers(
        self, req: AnalyzerRequest
    ) -> Union[AnswerGenerationInlineResult, AnswerGenerationFileResult]:
        if not isinstance(req.payload, AnswerGenerationRequest):
            raise ValueError("generate_answers requires AnswerGenerationRequest payload.")
        if not isinstance(req.input, DocumentPayload):
            raise ValueError("generate_answers requires DocumentPayload input.")

        expected = len(req.payload.questions)
        if expected < 1:
            raise ValueError("Questions list cannot be empty.")

        questions_text = "\n".join(req.payload.questions)
        content = generate_answers_text(
            req.input.text,
            questions_text=questions_text,
            expected_question_count=expected,
        )

        in_fmt = req.input.metadata.input_format
        if in_fmt == DocumentInputFormat.txt:
            return build_answer_generation_inline_result(
                content=content,
                expected_question_count=expected,
                algorithm_version=self.config.algorithm_version,
            )

        if in_fmt in (DocumentInputFormat.pdf, DocumentInputFormat.docx):
            out_fmt = FileOutputFormat.pdf if in_fmt == DocumentInputFormat.pdf else FileOutputFormat.docx
            filename = f"{_safe_basename(in_fmt.value)}.answers.{out_fmt.value}"

            written = write_document(
                content=content,
                output_format=out_fmt.value,
                output_name=filename,
            )
            return build_answer_generation_file_result(
                filename=written.file_name,
                output_format=out_fmt,
                file_size_mb=written.file_size_mb,
                expected_question_count=expected,
                algorithm_version=self.config.algorithm_version,
            )

        raise ValueError("generate_answers only supports input formats: pdf, docx, txt (strict v1 contract).")

    def _doc_action_inline_or_file(
        self,
        req: AnalyzerRequest,
        *,
        inline_builder,
        file_suffix: str,
    ) -> Union[InlineTextResult, FileResult]:
        if not isinstance(req.input, DocumentPayload):
            raise ValueError(f"{req.action.value} requires DocumentPayload input.")

        in_fmt = req.input.metadata.input_format
        text = req.input.text or ""
        if not text.strip():
            raise ValueError(f"{req.action.value} requires extracted document text.")

        if in_fmt == DocumentInputFormat.txt:
            content = inline_builder(text)
            return build_inline_txt_result(content=content, algorithm_version=self.config.algorithm_version)

        if in_fmt in (DocumentInputFormat.pdf, DocumentInputFormat.docx):
            out_fmt = FileOutputFormat.pdf if in_fmt == DocumentInputFormat.pdf else FileOutputFormat.docx
            filename = f"{_safe_basename(in_fmt.value)}.{file_suffix}.{out_fmt.value}"

            content = inline_builder(text)
            written = write_document(
                content=content,
                output_format=out_fmt.value,
                output_name=filename,
            )
            return build_file_result(
                filename=written.file_name,
                output_format=out_fmt,
                file_size_mb=written.file_size_mb,
                algorithm_version=self.config.algorithm_version,
            )

        raise ValueError(f"{req.action.value} only supports input formats: pdf, docx, txt (strict v1 contract).")

    def _build_response(
        self,
        req: AnalyzerRequest,
        *,
        result: Any,
        detected_language: Optional[str],
        output_language: Optional[str],
    ) -> AnalyzerResponse:
        if isinstance(req.input, MediaPayload):
            in_fmt: Union[DocumentInputFormat, str] = "audio" if req.input.media_type == MediaType.audio else "video"
        else:
            in_fmt = req.input.metadata.input_format

        dl = detected_language if self.config.include_language_fields else None
        ol = output_language if self.config.include_language_fields else None

        return AnalyzerResponse(
            action=req.action,
            input_format=in_fmt,
            policy=req.policy,
            system_language=req.system_language if isinstance(req.system_language, SystemLanguage) else SystemLanguage.english,
            detected_language=dl,
            output_language=ol,
            result=result,
        )

    def _maybe_language_fields_for_non_translate(self, req: AnalyzerRequest) -> tuple[Optional[str], Optional[str]]:
        if not self.config.include_language_fields:
            return None, None

        forced = _system_language_to_tag(req.system_language)
        return forced, forced

    def _language_fields_for_translate(self, req: AnalyzerRequest) -> tuple[Optional[str], Optional[str]]:
        if not self.config.include_language_fields:
            return None, None
        if not isinstance(req.payload, TranslationRequest):
            return None, None

        detected: Optional[str] = None
        if req.payload.source_language != "auto":
            detected = req.payload.source_language

        return detected, req.payload.target_language


def _system_language_to_tag(language: SystemLanguage) -> str:
    return "en" if language == SystemLanguage.english else "fr"


def _conversion_to_file_output(conv_value: str) -> FileOutputFormat:
    if conv_value == "pdf":
        return FileOutputFormat.pdf
    if conv_value == "docx":
        return FileOutputFormat.docx
    if conv_value == "jpg":
        return FileOutputFormat.jpg
    if conv_value == "jpeg":
        return FileOutputFormat.jpeg
    raise ValueError(f"Unsupported conversion output_format: {conv_value}")


def _safe_basename(ext_or_label: str) -> str:
    clean = "".join(ch for ch in ext_or_label if ch.isalnum() or ch in ("-", "_")).strip("-_")
    return clean or "document"


__all__ = ["Analyzer", "AnalyzerConfig"]

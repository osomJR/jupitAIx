from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


# =========================
# CONTRACT CONSTANTS (V1)
# =========================

MAX_FILE_SIZE_MB = 10
MAX_WORD_COUNT = 1000

MAX_AUDIO_SIZE_MB = 10
MAX_AUDIO_DURATION_SECONDS = 120  # 2 minutes

MAX_VIDEO_SIZE_MB = 25
MAX_VIDEO_DURATION_SECONDS = 180  # 3 minutes

ANONYMOUS_USER_MAX_ACTIONS_PER_DAY = 4
AUTHENTICATED_USER_MAX_ACTIONS_PER_DAY = 7


# =========================
# FORMATS (V1)
# =========================

class DocumentInputFormat(str, Enum):
    # Contract: .pdf, .docx, .txt, .jpg/.jpeg, .png
    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    jpg = "jpg"
    jpeg = "jpeg"
    png = "png"


class ConversionOutputFormat(str, Enum):
    # Contract conversion outputs only ever: pdf, docx, jpg, jpeg
    pdf = "pdf"
    docx = "docx"
    jpg = "jpg"
    jpeg = "jpeg"


class FileOutputFormat(str, Enum):
    # Contract: only file outputs are .pdf, .docx, .jpg/.jpeg
    pdf = "pdf"
    docx = "docx"
    jpg = "jpg"
    jpeg = "jpeg"


class InlineOutputFormat(str, Enum):
    # Contract: inline text output only
    txt = "txt"


class MediaType(str, Enum):
    audio = "audio"
    video = "video"


class AudioFormat(str, Enum):
    # Contract: audio upload .mp3
    mp3 = "mp3"


class VideoFormat(str, Enum):
    # Contract: video upload .mp4, .mkv, .mov
    mp4 = "mp4"
    mkv = "mkv"
    mov = "mov"


# =========================
# LANGUAGES
# =========================

class UILanguage(str, Enum):
    """
    UI language toggle. Requirement: English/French only.
    Affects UI chrome, error messages, and any guidance text for ALL features (including translate).
    """
    english = "english"
    french = "french"


class SystemLanguageMode(str, Enum):
    """
    Processing language mode.
    - For NON-translate features: 'auto' means detect English vs French only.
    - For translate: source_language controls source detection; system_language still applies to UI/guidance only.
    """
    auto = "auto"
    english = "english"
    french = "french"


BCP47Like = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=35,
        # Lightweight BCP-47-ish validation:
        # examples: en, fr, yo, de, pt-BR, zh-Hans, es-419
        pattern=r"^[A-Za-z]{2,3}([_-][A-Za-z0-9]{2,8})*$",
    ),
]


def _is_en_or_fr_tag(tag: str) -> bool:
    t = tag.lower().replace("_", "-")
    return t == "en" or t.startswith("en-") or t == "fr" or t.startswith("fr-")


def _system_mode_to_tag(mode: SystemLanguageMode) -> Optional[str]:
    if mode == SystemLanguageMode.english:
        return "en"
    if mode == SystemLanguageMode.french:
        return "fr"
    return None


# =========================
# FEATURE ENUM (V1)
# =========================

class FeatureType(str, Enum):
    convert = "convert"
    summarize = "summarize"
    grammar_correct = "grammar_correct"
    translate = "translate"
    transcribe = "transcribe"
    explain = "explain"
    generate_questions = "generate_questions"
    generate_answers = "generate_answers"


# =========================
# OUTPUT POLICY (V1)
# =========================

class OutputPolicy(BaseModel):
    # Contract: tone preservation ON, professional neutrality ON (non-optional)
    tone_preservation: Literal[True] = True
    professional_neutrality: Literal[True] = True

    # Contract: structure preservation ON for transformed outputs (convert/summarize/grammar/translate/transcribe)
    # and OFF for generated outputs (explain/questions/answers).
    structure_preservation: bool = Field(
        ...,
        description="True for transformed outputs; False for generated outputs (per contract).",
    )


# =========================
# INPUT ARTIFACTS
# =========================

class DocumentMetadata(BaseModel):
    input_format: DocumentInputFormat
    file_size_mb: float = Field(..., ge=0)

    # Word-count post extraction (<= 1000). Required ONLY for AI document actions (see AnalyzerRequest validator).
    extracted_word_count: Optional[int] = Field(default=None, ge=0, le=MAX_WORD_COUNT)

    # OCR enabled when needed. Must be False for docx/txt.
    ocr_used: bool = False

    # Detected language (optional carrier; typically supplied by server-side detection).
    # - For NON-translate features, validated to en*/fr* when present.
    # - For translate, can be any BCP-47-like tag if you choose to populate it.
    detected_language: Optional[BCP47Like] = None

    @field_validator("ocr_used")
    @classmethod
    def validate_ocr_used(cls, v: bool, info):
        fmt = info.data.get("input_format")
        if fmt in (DocumentInputFormat.docx, DocumentInputFormat.txt) and v:
            raise ValueError("ocr_used must be False for docx/txt inputs.")
        return v


class DocumentPayload(BaseModel):
    # For AI-processing actions, extracted text is required.
    # For conversion actions, text can be omitted.
    text: Optional[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = None
    metadata: DocumentMetadata

    # Real persisted file reference (set by upload layer)
    filename: Optional[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = None


class MediaPayload(BaseModel):
    media_type: MediaType
    media_format: Union[AudioFormat, VideoFormat]
    file_size_mb: float = Field(..., ge=0)
    duration_seconds: int = Field(..., ge=1)

    filename: Optional[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = None
    mime_type: Optional[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = None

    # Detected language (optional carrier; typically supplied by server-side detection).
    # For transcribe, validated to en*/fr* when present.
    detected_language: Optional[BCP47Like] = None

    @model_validator(mode="after")
    def validate_media_limits_and_format(self):
        if self.media_type == MediaType.audio:
            if self.media_format != AudioFormat.mp3:
                raise ValueError("Audio media_format must be mp3.")
            if self.file_size_mb > MAX_AUDIO_SIZE_MB:
                raise ValueError(f"Audio size must be <= {MAX_AUDIO_SIZE_MB} MB.")
            if self.duration_seconds > MAX_AUDIO_DURATION_SECONDS:
                raise ValueError(f"Audio duration must be <= {MAX_AUDIO_DURATION_SECONDS} seconds.")
        else:
            if not isinstance(self.media_format, VideoFormat):
                raise ValueError("Video media_format must be one of: mp4, mkv, mov.")
            if self.file_size_mb > MAX_VIDEO_SIZE_MB:
                raise ValueError(f"Video size must be <= {MAX_VIDEO_SIZE_MB} MB.")
            if self.duration_seconds > MAX_VIDEO_DURATION_SECONDS:
                raise ValueError(f"Video duration must be <= {MAX_VIDEO_DURATION_SECONDS} seconds.")
        return self


InputArtifact = Union[DocumentPayload, MediaPayload]


# =========================
# FEATURE PAYLOADS
# =========================

class ConversionRequest(BaseModel):
    feature: Literal[FeatureType.convert]
    output_format: ConversionOutputFormat

    @staticmethod
    def _allowed_conversion_pairs() -> set[tuple[DocumentInputFormat, ConversionOutputFormat]]:
        # Contract conversion set ONLY:
        # - PDF <-> Word
        # - Image (jpg/jpeg) -> PDF / Word
        # - Image (png) -> Image (jpg/jpeg)
        return {
            (DocumentInputFormat.pdf, ConversionOutputFormat.docx),
            (DocumentInputFormat.docx, ConversionOutputFormat.pdf),
            (DocumentInputFormat.jpg, ConversionOutputFormat.pdf),
            (DocumentInputFormat.jpg, ConversionOutputFormat.docx),
            (DocumentInputFormat.jpeg, ConversionOutputFormat.pdf),
            (DocumentInputFormat.jpeg, ConversionOutputFormat.docx),
            (DocumentInputFormat.png, ConversionOutputFormat.jpg),
            (DocumentInputFormat.png, ConversionOutputFormat.jpeg),
        }

    def validate_pair(self, input_format: DocumentInputFormat) -> None:
        if (input_format, self.output_format) not in self._allowed_conversion_pairs():
            raise ValueError(
                f"Unsupported conversion pair: {input_format.value} -> {self.output_format.value} (strict v1 contract)."
            )


class SummarizationRequest(BaseModel):
    feature: Literal[FeatureType.summarize]


class GrammarCorrectionRequest(BaseModel):
    feature: Literal[FeatureType.grammar_correct]


class TranslationRequest(BaseModel):
    """
    Translate only:
    - supports any source language -> any target language
    - supports auto source detection via source_language="auto"
    UI language (ui_language) still controls errors/messages/guidance language.
    """
    feature: Literal[FeatureType.translate]
    source_language: Literal["auto"] | BCP47Like = "auto"
    target_language: BCP47Like


class TranscriptionRequest(BaseModel):
    feature: Literal[FeatureType.transcribe]
    preserve_filler_words: bool = True
    remove_background_noise: bool = False
    diarize_speakers: bool = True


class ExplanationRequest(BaseModel):
    feature: Literal[FeatureType.explain]
    allow_external_knowledge: bool = False


class QuestionGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_questions]
    allow_external_knowledge: Literal[False] = False


class AnswerGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_answers]
    allow_external_knowledge: Literal[False] = False
    questions: List[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]]

    @field_validator("questions")
    @classmethod
    def validate_numbered_questions(cls, v: List[str]):
        if not v:
            raise ValueError("Questions list cannot be empty.")
        for i, q in enumerate(v, start=1):
            if not q.lstrip().startswith(f"{i}."):
                raise ValueError("Questions must be sequentially numbered starting at 1 (e.g., '1. ...').")
        return v


FeaturePayload = Union[
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    TranscriptionRequest,
    ExplanationRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
]


# =========================
# QUESTION COUNT SCALING (V1)
# =========================

class QuestionScale(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class QuestionScalingRule(BaseModel):
    classification: QuestionScale
    min_words: int
    max_words: int
    min_questions: int
    max_questions: int

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.max_words < self.min_words:
            raise ValueError("Invalid word range: max_words must be >= min_words.")
        if self.max_questions < self.min_questions:
            raise ValueError("Invalid question range: max_questions must be >= min_questions.")
        return self


QUESTION_SCALING_RULES: List[QuestionScalingRule] = [
    QuestionScalingRule(classification=QuestionScale.small, min_words=1, max_words=300, min_questions=4, max_questions=6),
    QuestionScalingRule(classification=QuestionScale.medium, min_words=301, max_words=700, min_questions=8, max_questions=10),
    QuestionScalingRule(classification=QuestionScale.large, min_words=701, max_words=1000, min_questions=12, max_questions=15),
]


def classify_word_count(word_count: int) -> QuestionScalingRule:
    for rule in QUESTION_SCALING_RULES:
        if rule.min_words <= word_count <= rule.max_words:
            return rule
    raise ValueError(f"Word count {word_count} is out of supported range (1-1000).")


# =========================
# REQUEST ENVELOPE + CONTRACT ENFORCEMENT
# =========================

AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT = {
    # extracted_word_count required ONLY for these actions (upload or inline text)
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

NON_TRANSLATE_DOC_ACTIONS = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

_CONVERSION_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

_AI_DOC_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.txt,
}


class AnalyzerRequest(BaseModel):
    """
    - ui_language: English/French UI chrome + errors/messages for ALL features (including translate).
    - system_language: for NON-translate features, auto-detect English vs French; or force English/French.
      For translate, source_language/target_language determine translation direction; ui_language controls UI messaging.

    NOTE (server supplies detection):
    - detected_language fields are optional in requests; server performs detection at runtime.
    - When system_language='auto' for non-translate features (and transcribe), the server MUST reject if detected language is not en*/fr*.
    """
    action: FeatureType
    input: InputArtifact
    payload: FeaturePayload
    policy: OutputPolicy

    ui_language: UILanguage = UILanguage.english
    system_language: SystemLanguageMode = SystemLanguageMode.auto

    @model_validator(mode="after")
    def validate_contract_rules(self):
        # action must match payload.feature
        if self.payload.feature != self.action:
            raise ValueError("action must match payload.feature exactly.")

        # input type enforcement
        if self.action == FeatureType.transcribe:
            if not isinstance(self.input, MediaPayload):
                raise ValueError("transcribe requires MediaPayload as input.")
        else:
            if not isinstance(self.input, DocumentPayload):
                raise ValueError(f"{self.action.value} requires DocumentPayload as input.")
        # reject client-supplied detected_language (server supplies detection)
        if isinstance(self.input, DocumentPayload) and self.input.metadata.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection."
    )
        if isinstance(self.input, MediaPayload) and self.input.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection."
    )

        # per-feature input extension enforcement (contract)
        if self.action in AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _AI_DOC_INPUTS:
                raise ValueError(
                    f"{self.action.value} only supports input formats: pdf, docx, txt (strict v1 contract)."
                )

        # (1) Convert is language-agnostic (no language gating). It is a format transformation feature.
        if self.action == FeatureType.convert:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _CONVERSION_INPUTS:
                raise ValueError("convert only supports: pdf, docx, jpg, jpeg, png (strict v1 contract).")
            assert isinstance(self.payload, ConversionRequest)
            self.payload.validate_pair(self.input.metadata.input_format)

        # extracted text + extracted_word_count required ONLY for AI document actions,
        # never required for conversion and transcription.
        if self.action in AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            assert isinstance(self.input, DocumentPayload)
            if not self.input.text:
                raise ValueError(f"{self.action.value} requires extracted document text.")
            if self.input.metadata.extracted_word_count is None:
                raise ValueError(f"{self.action.value} requires extracted_word_count.")
            if self.input.metadata.extracted_word_count < 1:
                raise ValueError("extracted_word_count must be >= 1 for AI processing actions.")
            if self.input.metadata.extracted_word_count > MAX_WORD_COUNT:
                raise ValueError(
            f"extracted_word_count must be <= {MAX_WORD_COUNT} for AI processing actions."
        )

        # structure preservation policy enforcement (contract)
        transformed = {
            FeatureType.convert,
            FeatureType.summarize,
            FeatureType.grammar_correct,
            FeatureType.translate,
            FeatureType.transcribe,
        }
        generated = {
            FeatureType.explain,
            FeatureType.generate_questions,
            FeatureType.generate_answers,
        }
        if self.action in transformed and self.policy.structure_preservation is not True:
            raise ValueError("structure_preservation must be True for transformed outputs (strict v1 contract).")
        if self.action in generated and self.policy.structure_preservation is not False:
            raise ValueError("structure_preservation must be False for generated outputs (strict v1 contract).")

        # (2) Auto-detection rules (request-side validation, server-supplied detection):
        # - Non-translate features: system_language="auto" expresses “detect English vs French”.
        #   If detected_language is supplied, validate en*/fr*.
        # - Translate: source_language="auto" expresses “detect any source language”, and target supports any language tag.
        if self.action in NON_TRANSLATE_DOC_ACTIONS:
            assert isinstance(self.input, DocumentPayload)

            forced_tag = _system_mode_to_tag(self.system_language)

            # If detected_language is supplied, it must be en*/fr* for non-translate features.
            if self.input.metadata.detected_language is not None and not _is_en_or_fr_tag(self.input.metadata.detected_language):
                raise ValueError(
                    "Non-translate features support only English/French. detected_language must be en*/fr* when provided."
                )

            # Optional consistency check between forced system_language and detected_language
            if forced_tag and self.input.metadata.detected_language:
                d = self.input.metadata.detected_language.lower().replace("_", "-")
                if forced_tag == "en" and not (d == "en" or d.startswith("en-")):
                    raise ValueError("system_language='english' but detected_language is not en*.")
                if forced_tag == "fr" and not (d == "fr" or d.startswith("fr-")):
                    raise ValueError("system_language='french' but detected_language is not fr*.")

        if self.action == FeatureType.transcribe:
            # Transcribe output is restricted to English/French (runtime enforcement by server).
            # Request-side: if detected_language is supplied, validate en*/fr* and optionally validate against forced system_language.
            forced_tag = _system_mode_to_tag(self.system_language)
            assert isinstance(self.input, MediaPayload)

            if self.input.detected_language is not None and not _is_en_or_fr_tag(self.input.detected_language):
                raise ValueError("transcribe supports only English/French (detected_language must be en*/fr* when provided).")

            if forced_tag and self.input.detected_language:
                d = self.input.detected_language.lower().replace("_", "-")
                if forced_tag == "en" and not (d == "en" or d.startswith("en-")):
                    raise ValueError("system_language='english' but detected_language is not en*.")
                if forced_tag == "fr" and not (d == "fr" or d.startswith("fr-")):
                    raise ValueError("system_language='french' but detected_language is not fr*.")

        # Translate is the only feature that supports any source/target language;
        # its detection is governed by TranslationRequest.source_language and TranslationRequest.target_language typing.
        return self


# =========================
# RESPONSE MODELS (V1)
# =========================

class DeterminismMetadata(BaseModel):
    deterministic: Literal[True] = True
    contract_version: Literal["v1"] = "v1"
    algorithm_version: Optional[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = None


class InlineTextResult(BaseModel):
    output_format: Literal[InlineOutputFormat.txt] = InlineOutputFormat.txt
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    meta: DeterminismMetadata


class FileResult(BaseModel):
    filename: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    output_format: FileOutputFormat
    file_size_mb: float = Field(..., ge=0)
    meta: DeterminismMetadata


class QuestionScaleMetadata(BaseModel):
    classification: QuestionScale
    extracted_word_count: int = Field(..., ge=1, le=MAX_WORD_COUNT)


class QuestionGenerationInlineResult(InlineTextResult):
    scale: QuestionScaleMetadata

    @model_validator(mode="after")
    def enforce_question_scaling(self):
        rule = classify_word_count(self.scale.extracted_word_count)
        if self.scale.classification != rule.classification:
            raise ValueError("classification mismatch for extracted_word_count.")
        lines = [ln.strip() for ln in self.content.splitlines() if ln.strip()]
        numbered = [ln for ln in lines if ln[:1].isdigit() and "." in ln.split()[0]]
        n = len(numbered) if numbered else 0
        if not (rule.min_questions <= n <= rule.max_questions):
            raise ValueError(
                f"Question count out of range for {rule.classification.value}: expected {rule.min_questions}–{rule.max_questions}."
            )
        for i in range(1, n + 1):
            if not any(ln.startswith(f"{i}.") for ln in numbered):
                raise ValueError("Questions must be sequentially numbered starting at 1.")
        return self


class QuestionGenerationFileResult(FileResult):
    scale: QuestionScaleMetadata


class AnswerGenerationInlineResult(InlineTextResult):
    expected_question_count: int = Field(..., ge=1)

    @model_validator(mode="after")
    def enforce_answer_alignment(self):
        lines = [ln.strip() for ln in self.content.splitlines() if ln.strip()]
        numbered = [ln for ln in lines if ln[:1].isdigit() and "." in ln.split()[0]]
        n = len(numbered) if numbered else 0
        if n != self.expected_question_count:
            raise ValueError("Answer count must exactly match the number of questions.")
        for i in range(1, n + 1):
            if not any(ln.startswith(f"{i}.") for ln in numbered):
                raise ValueError("Answers must be sequentially numbered starting at 1.")
        return self


class AnswerGenerationFileResult(FileResult):
    expected_question_count: int = Field(..., ge=1)


AnalyzerResult = Union[
    InlineTextResult,
    FileResult,
    QuestionGenerationInlineResult,
    QuestionGenerationFileResult,
    AnswerGenerationInlineResult,
    AnswerGenerationFileResult,
]


class AnalyzerResponse(BaseModel):
    action: FeatureType
    input_format: Union[DocumentInputFormat, Literal["audio"], Literal["video"]]
    policy: OutputPolicy

    # UI chrome language (English/French only) — applies to ALL features, including translate.
    ui_language: UILanguage = UILanguage.english

    # Processing language mode for non-translate features.
    system_language: SystemLanguageMode = SystemLanguageMode.auto

    # Detected input language (optional echo). For non-translate features, en*/fr* when present.
    detected_language: Optional[BCP47Like] = None

    # Explicit output language marker (useful to enforce/observe keep-same-language at the API boundary):
    # - For non-translate features: when provided must match en*/fr* and SHOULD equal detected input language.
    # - For translate: can be any tag and SHOULD equal the requested target_language.
    output_language: Optional[BCP47Like] = None

    result: AnalyzerResult

    @model_validator(mode="after")
    def validate_action_result_mapping_and_output_rules(self):
        # 1) Action-result family constraints
        if self.action == FeatureType.convert:
            if not isinstance(self.result, FileResult):
                raise ValueError("convert must return a file result.")
        elif self.action == FeatureType.transcribe:
            if not isinstance(self.result, InlineTextResult):
                raise ValueError("transcribe must return inline txt only (strict v1 contract).")
        elif self.action in {
            FeatureType.summarize,
            FeatureType.grammar_correct,
            FeatureType.translate,
            FeatureType.explain,
        }:
            if not isinstance(self.result, (InlineTextResult, FileResult)):
                raise ValueError(f"{self.action.value} must return inline txt or a file result.")
        elif self.action == FeatureType.generate_questions:
            if not isinstance(self.result, (QuestionGenerationInlineResult, QuestionGenerationFileResult)):
                raise ValueError("generate_questions must return question-generation result.")
        elif self.action == FeatureType.generate_answers:
            if not isinstance(self.result, (AnswerGenerationInlineResult, AnswerGenerationFileResult)):
                raise ValueError("generate_answers must return answer-generation result.")

        # 2) Output-extension rules for AI doc actions (strict):
        # - input txt -> inline txt
        # - input pdf/docx -> file with same extension
        if self.action in AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            if isinstance(self.input_format, DocumentInputFormat):
                if self.input_format == DocumentInputFormat.txt:
                    if not isinstance(self.result, InlineTextResult):
                        raise ValueError("For txt input, output must be inline txt (strict v1 contract).")
                    if self.result.output_format != InlineOutputFormat.txt:
                        raise ValueError("Inline output must be txt.")
                elif self.input_format in (DocumentInputFormat.pdf, DocumentInputFormat.docx):
                    if not isinstance(self.result, FileResult):
                        raise ValueError("For pdf/docx input, output must be a downloadable file (strict v1 contract).")
                    expected = FileOutputFormat.pdf if self.input_format == DocumentInputFormat.pdf else FileOutputFormat.docx
                    if self.result.output_format != expected:
                        raise ValueError("Output file extension must match input extension (strict v1 contract).")
                else:
                    raise ValueError("AI doc actions only support input formats: pdf, docx, txt.")
            else:
                raise ValueError("AI doc actions require a document input_format.")

        # 3) Transcription output rule: always inline txt
        if self.action == FeatureType.transcribe:
            if not isinstance(self.result, InlineTextResult) or self.result.output_format != InlineOutputFormat.txt:
                raise ValueError("transcribe output must be inline txt (strict v1 contract).")

        # 4) Language boundary validation for non-translate features (when language fields are provided)
        if self.action in NON_TRANSLATE_DOC_ACTIONS or self.action == FeatureType.transcribe:
            if self.detected_language is not None and not _is_en_or_fr_tag(self.detected_language):
                raise ValueError("Non-translate responses must have detected_language en*/fr* when provided.")
            if self.output_language is not None and not _is_en_or_fr_tag(self.output_language):
                raise ValueError("Non-translate responses must have output_language en*/fr* when provided.")
            # Keep same language as input:
            if self.detected_language and self.output_language:
                d = self.detected_language.lower().replace("_", "-")
                o = self.output_language.lower().replace("_", "-")
                if (d.startswith("en") and not o.startswith("en")) or (d.startswith("fr") and not o.startswith("fr")):
                    raise ValueError(
                        "For non-translate features, output_language must match detected_language (keep same language)."
                    )

        return self
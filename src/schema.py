from __future__ import annotations
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

# CONTRACT CONSTANTS (V1)

MAX_FILE_SIZE_MB = 10
MAX_WORD_COUNT = 1000
MAX_AUDIO_SIZE_MB = 10
MAX_AUDIO_DURATION_SECONDS = 120  # 2 minutes
MAX_VIDEO_SIZE_MB = 25
MAX_VIDEO_DURATION_SECONDS = 180  # 3 minutes

# FORMATS (V1)

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
    
    # Contract: file outputs are .pdf, .docx, .jpg/.jpeg, .png
    
    pdf = "pdf"
    docx = "docx"
    jpg = "jpg"
    jpeg = "jpeg"
    png = "png"

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

# LANGUAGES

class SystemLanguage(str, Enum):
    """
    Backend processing language selection for supported product languages.
    Frontend UI language must stay synchronized with this value.
    Backend does not model UI language separately.
    """
    english = "english"
    french = "french"

BCP47Like = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=35,
        # examples: en, fr, yo, de, pt-BR, zh-Hans, es-419
        pattern=r"^[A-Za-z]{2,3}([_-][A-Za-z0-9]{2,8})*$",
    ),
]

def _is_en_or_fr_tag(tag: str) -> bool:
    t = tag.lower().replace("_", "-")
    return t == "en" or t.startswith("en-") or t == "fr" or t.startswith("fr-")

def _system_language_to_tag(language: SystemLanguage) -> str:
    return "en" if language == SystemLanguage.english else "fr"

# FEATURE ENUM (V1)

class FeatureType(str, Enum):
    convert = "convert"
    summarize = "summarize"
    grammar_correct = "grammar_correct"
    translate = "translate"
    transcribe = "transcribe"
    explain = "explain"
    redact = "redact"
    data_mask = "data_mask"
    generate_questions = "generate_questions"
    generate_answers = "generate_answers"

# OUTPUT POLICY (V1)

class OutputPolicy(BaseModel):
   
    # Contract: tone preservation ON, professional neutrality ON (non-optional)
   
    tone_preservation: Literal[True] = True
    professional_neutrality: Literal[True] = True

    # Contract: structure preservation ON for transformed outputs
    # (convert/summarize/grammar/translate/transcribe/redact/data_mask)
    # and OFF for generated outputs (explain/questions/answers).
    
    structure_preservation: bool = Field(
        ...,
        description="True for transformed outputs; False for generated outputs (per contract).",
    )

# INPUT ARTIFACTS

class DocumentMetadata(BaseModel):
    input_format: DocumentInputFormat
    file_size_mb: float = Field(..., ge=0, le=MAX_FILE_SIZE_MB)

    # Word-count post extraction (<= 1000). Required for text-based AI document actions,
    # and also for redaction/data masking word-limit enforcement.
    
    extracted_word_count: Optional[int] = Field(default=None, ge=0, le=MAX_WORD_COUNT)

    # OCR enabled when needed. Must be False for docx/txt.
    
    ocr_used: bool = False

    # Server-side detected language (do not accept from client requests).
    
    detected_language: Optional[BCP47Like] = None

    @field_validator("ocr_used")
    @classmethod
    def validate_ocr_used(cls, v: bool, info):
        fmt = info.data.get("input_format")
        if fmt in (DocumentInputFormat.docx, DocumentInputFormat.txt) and v:
            raise ValueError("ocr_used must be False for docx/txt inputs.")
        return v

class DocumentPayload(BaseModel):
    
    # For text-based AI-processing actions, extracted text is required.
    # For conversion, redaction, and data masking, text can be omitted.
    
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

    # Server-side detected language (do not accept from client requests).
    
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

# FEATURE PAYLOADS

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
    Translation adjustment requested by product owner:
    - supports any source language -> any target language, as long as the underlying LLM/API supports it
    - supports auto source detection via source_language="auto"
    Note:
    - system_language remains English/French only and is intended to stay synchronized with frontend UI language.
    - translation direction is governed exclusively by source_language/target_language.
    """
    feature: Literal[FeatureType.translate]
    source_language: Literal["auto"] | BCP47Like = "auto"
    target_language: BCP47Like
    @field_validator("target_language")
    @classmethod
    def validate_target_language(cls, v: str):
        if v.lower() == "auto":
            raise ValueError("target_language cannot be 'auto'.")
        return v

class TranscriptionRequest(BaseModel):
    feature: Literal[FeatureType.transcribe]
    preserve_filler_words: bool = True
    remove_background_noise: bool = False
    diarize_speakers: bool = True

class ExplanationRequest(BaseModel):
    feature: Literal[FeatureType.explain]
    allow_external_knowledge: bool = False

class SensitiveDataType(str, Enum):
    name = "name"
    email_address = "email_address"
    phone_number = "phone_number"
    account_number = "account_number"
    card_number = "card_number"
    national_id = "national_id"
    tax_id = "tax_id"
    passport_number = "passport_number"
    contact_address = "contact_address"
    date_of_birth = "date_of_birth"
    age = "age"
    signature = "signature"

class RedactionMaskingDocumentType(str, Enum):
    invoice = "invoice"
    kyc_document = "kyc_document"
    bank_statement = "bank_statement"
    contract = "contract"
    id_document = "id_document"
    legal_document = "legal_document"

_ALL_SENSITIVE_DATA_TYPES = [item for item in SensitiveDataType]

class RedactionRequest(BaseModel):
    feature: Literal[FeatureType.redact]
    document_type: Optional[RedactionMaskingDocumentType] = None
    target_data: List[SensitiveDataType] = Field(default_factory=lambda: list(_ALL_SENSITIVE_DATA_TYPES))
    review_exclusions: List[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = Field(
        default_factory=list,
        description="Items the user reviewed and explicitly chose not to redact before final export.",
    )

class DataMaskingRequest(BaseModel):
    feature: Literal[FeatureType.data_mask]
    document_type: Optional[RedactionMaskingDocumentType] = None
    target_data: List[SensitiveDataType] = Field(default_factory=lambda: list(_ALL_SENSITIVE_DATA_TYPES))
    review_exclusions: List[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = Field(
        default_factory=list,
        description="Items the user reviewed and explicitly chose not to mask before final export.",
    )
    masking_mode: Literal["partial"] = "partial"

class QuestionGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_questions]
    allow_external_knowledge: Literal[False] = False

class AnswerGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_answers]
    allow_external_knowledge: Literal[False] = False

    # Encodes the product rule that answer generation is only a follow-on action
    # after question generation. The frontend must still enforce button visibility.
    
    prerequisite_action: Literal[FeatureType.generate_questions] = FeatureType.generate_questions
    requires_generated_questions: Literal[True] = True
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
    RedactionRequest,
    DataMaskingRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
]

# QUESTION COUNT SCALING (V1)

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

# REQUEST ENVELOPE + CONTRACT ENFORCEMENT

TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

DOC_ACTIONS_REQUIRING_WORDCOUNT_ONLY = {
    FeatureType.redact,
    FeatureType.data_mask,
}

EN_FR_ONLY_DOC_ACTIONS = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.explain,
    FeatureType.redact,
    FeatureType.data_mask,
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

_TEXT_AI_DOC_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.txt,
}

_REDACTION_MASKING_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

class AnalyzerRequest(BaseModel):
    """
    Backend request contract.

    - No ui_language field exists in the backend schema.
    - system_language is English/French only and should be kept synchronized by the frontend
      with the UI language selection.
    - Translation source/target selection is handled inside TranslationRequest.
    NOTE (server supplies detection):
    - detected_language fields are optional carriers and are not accepted from external clients.
    - server performs detection at runtime.
    """
    action: FeatureType
    input: InputArtifact
    payload: FeaturePayload
    policy: OutputPolicy
    system_language: SystemLanguage = SystemLanguage.english
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
            raise ValueError("detected_language must not be provided by client; server supplies detection.")
        if isinstance(self.input, MediaPayload) and self.input.detected_language is not None:
            raise ValueError("detected_language must not be provided by client; server supplies detection.")

        # per-feature input extension enforcement
        
        if self.action in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _TEXT_AI_DOC_INPUTS:
                raise ValueError(
                    f"{self.action.value} only supports input formats: pdf, docx, txt (strict contract rule)."
                )
        if self.action in DOC_ACTIONS_REQUIRING_WORDCOUNT_ONLY:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _REDACTION_MASKING_INPUTS:
                raise ValueError(
                    f"{self.action.value} only supports input formats: pdf, docx, jpg, jpeg, png (strict contract rule)."
                )
        if self.action == FeatureType.convert:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _CONVERSION_INPUTS:
                raise ValueError("convert only supports: pdf, docx, jpg, jpeg, png (strict contract rule).")
            assert isinstance(self.payload, ConversionRequest)
            self.payload.validate_pair(self.input.metadata.input_format)

        # text + word count required for text-based AI document actions
        
        if self.action in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            assert isinstance(self.input, DocumentPayload)
            if not self.input.text:
                raise ValueError(f"{self.action.value} requires extracted document text.")
            if self.input.metadata.extracted_word_count is None:
                raise ValueError(f"{self.action.value} requires extracted_word_count.")
            if self.input.metadata.extracted_word_count < 1:
                raise ValueError("extracted_word_count must be >= 1 for text-based AI processing actions.")
            if self.input.metadata.extracted_word_count > MAX_WORD_COUNT:
                raise ValueError(
                    f"extracted_word_count must be <= {MAX_WORD_COUNT} for text-based AI processing actions."
                )

        # word-count enforcement for redaction/data masking document requests
        
        if self.action in DOC_ACTIONS_REQUIRING_WORDCOUNT_ONLY:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.extracted_word_count is None:
                raise ValueError(f"{self.action.value} requires extracted_word_count for document limit enforcement.")
            if self.input.metadata.extracted_word_count > MAX_WORD_COUNT:
                raise ValueError(
                    f"extracted_word_count must be <= {MAX_WORD_COUNT} for {self.action.value}."
                )

        # structure preservation policy enforcement
       
        transformed = {
            FeatureType.convert,
            FeatureType.summarize,
            FeatureType.grammar_correct,
            FeatureType.translate,
            FeatureType.transcribe,
            FeatureType.redact,
            FeatureType.data_mask,
        }
        generated = {
            FeatureType.explain,
            FeatureType.generate_questions,
            FeatureType.generate_answers,
        }
        if self.action in transformed and self.policy.structure_preservation is not True:
            raise ValueError("structure_preservation must be True for transformed outputs (strict contract rule).")
        if self.action in generated and self.policy.structure_preservation is not False:
            raise ValueError("structure_preservation must be False for generated outputs (strict contract rule).")
        return self

# RESPONSE MODELS (V1)

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
    storage_key: Optional[str] = None
    download_url: Optional[str] = None
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

    # Backend-only processing language. Frontend UI language should stay synchronized with this value.
    
    system_language: SystemLanguage = SystemLanguage.english

    # Detected input language (optional echo).
    # - For non-translate features: en*/fr* when provided.
    # - For translate: any supported source tag when provided.
    
    detected_language: Optional[BCP47Like] = None

    # Explicit output language marker.
    # - For non-translate features: en*/fr* when provided and should match detected_language.
    # - For translate: any supported target tag when provided.
    
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
                raise ValueError("transcribe must return inline txt only (strict contract rule).")
        elif self.action in {
            FeatureType.summarize,
            FeatureType.grammar_correct,
            FeatureType.translate,
            FeatureType.explain,
        }:
            if not isinstance(self.result, (InlineTextResult, FileResult)):
                raise ValueError(f"{self.action.value} must return inline txt or a file result.")
        elif self.action in {FeatureType.redact, FeatureType.data_mask}:
            if not isinstance(self.result, FileResult):
                raise ValueError(f"{self.action.value} must return a file result.")
        elif self.action == FeatureType.generate_questions:
            if not isinstance(self.result, (QuestionGenerationInlineResult, QuestionGenerationFileResult)):
                raise ValueError("generate_questions must return question-generation result.")
        elif self.action == FeatureType.generate_answers:
            if not isinstance(self.result, (AnswerGenerationInlineResult, AnswerGenerationFileResult)):
                raise ValueError("generate_answers must return answer-generation result.")

        # 2) Output-extension rules
        # - text AI document actions: txt input -> inline txt; pdf/docx input -> same extension file
        # - redaction/data masking: pdf/docx/jpg/jpeg/png input -> same extension file
        
        if self.action in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            if isinstance(self.input_format, DocumentInputFormat):
                if self.input_format == DocumentInputFormat.txt:
                    if not isinstance(self.result, InlineTextResult):
                        raise ValueError("For txt input, output must be inline txt (strict contract rule).")
                    if self.result.output_format != InlineOutputFormat.txt:
                        raise ValueError("Inline output must be txt.")
                elif self.input_format in (DocumentInputFormat.pdf, DocumentInputFormat.docx):
                    if not isinstance(self.result, FileResult):
                        raise ValueError("For pdf/docx input, output must be a downloadable file (strict contract rule).")
                    expected = FileOutputFormat.pdf if self.input_format == DocumentInputFormat.pdf else FileOutputFormat.docx
                    if self.result.output_format != expected:
                        raise ValueError("Output file extension must match input extension (strict contract rule).")
                else:
                    raise ValueError("Text AI document actions only support input formats: pdf, docx, txt.")
            else:
                raise ValueError("Text AI document actions require a document input_format.")
        if self.action in {FeatureType.redact, FeatureType.data_mask}:
            if not isinstance(self.input_format, DocumentInputFormat):
                raise ValueError(f"{self.action.value} requires a document input_format.")
            if self.input_format == DocumentInputFormat.txt:
                raise ValueError(f"{self.action.value} does not support txt input.")
            if not isinstance(self.result, FileResult):
                raise ValueError(f"{self.action.value} output must be a downloadable file.")
            expected_map = {
                DocumentInputFormat.pdf: FileOutputFormat.pdf,
                DocumentInputFormat.docx: FileOutputFormat.docx,
                DocumentInputFormat.jpg: FileOutputFormat.jpg,
                DocumentInputFormat.jpeg: FileOutputFormat.jpeg,
                DocumentInputFormat.png: FileOutputFormat.png,
            }
            expected = expected_map[self.input_format]
            if self.result.output_format != expected:
                raise ValueError("Output file extension must match input extension (strict contract rule).")

        # 3) Transcription output rule: always inline txt
        
        if self.action == FeatureType.transcribe:
            if not isinstance(self.result, InlineTextResult) or self.result.output_format != InlineOutputFormat.txt:
                raise ValueError("transcribe output must be inline txt (strict contract rule).")

        # 4) Language boundary validation for non-translate features (when language fields are provided)
        
        if self.action in EN_FR_ONLY_DOC_ACTIONS or self.action == FeatureType.transcribe:
            if self.detected_language is not None and not _is_en_or_fr_tag(self.detected_language):
                raise ValueError("Non-translate responses must have detected_language en*/fr* when provided.")
            if self.output_language is not None and not _is_en_or_fr_tag(self.output_language):
                raise ValueError("Non-translate responses must have output_language en*/fr* when provided.")
            if self.detected_language and self.output_language:
                d = self.detected_language.lower().replace("_", "-")
                o = self.output_language.lower().replace("_", "-")
                if (d.startswith("en") and not o.startswith("en")) or (d.startswith("fr") and not o.startswith("fr")):
                    raise ValueError(
                        "For non-translate features, output_language must match detected_language (keep same language)."
                    )

        # 5) Optional consistency check between selected system_language and response language on non-translate actions
        
        if self.action in EN_FR_ONLY_DOC_ACTIONS or self.action == FeatureType.transcribe:
            forced_tag = _system_language_to_tag(self.system_language)
            if self.output_language is not None:
                o = self.output_language.lower().replace("_", "-")
                if forced_tag == "en" and not (o == "en" or o.startswith("en-")):
                    raise ValueError("system_language='english' but output_language is not en*.")
                if forced_tag == "fr" and not (o == "fr" or o.startswith("fr-")):
                    raise ValueError("system_language='french' but output_language is not fr*.")
        return self


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

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

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


class DocumentFileOutputFormat(str, Enum):
    # Contract: file outputs can be .pdf, .docx, .jpg/.jpeg, .png
    pdf = "pdf"
    docx = "docx"
    jpg = "jpg"
    jpeg = "jpeg"
    png = "png"


class StructuredDataOutputFormat(str, Enum):
    # Contract structured extraction outputs: .csv, .json, .xlsx
    csv = "csv"
    json = "json"
    xlsx = "xlsx"


class ComplianceOutputFormat(str, Enum):
    # Contract compliance outputs: human-readable report (.pdf),
    # machine-readable report (.json), annotated source output (.pdf)
    pdf = "pdf"
    json = "json"


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
    structured_extract = "structured_extract"
    compliance = "compliance"
    generate_questions = "generate_questions"
    generate_answers = "generate_answers"


# OUTPUT POLICY (V1)


class OutputPolicy(BaseModel):
    # Contract: tone preservation ON, professional neutrality ON (non-optional)
    tone_preservation: Literal[True] = True
    professional_neutrality: Literal[True] = True

    # Contract:
    # - True for transformed outputs:
    #   convert / summarize / grammar_correct / translate / transcribe / redact / data_mask
    # - False for generated outputs:
    #   explain / structured_extract / compliance / generate_questions / generate_answers
    structure_preservation: bool = Field(
        ...,
        description="True for transformed outputs; False for generated outputs (per contract).",
    )


# INPUT ARTIFACTS


class DocumentMetadata(BaseModel):
    input_format: DocumentInputFormat
    file_size_mb: float = Field(..., ge=0, le=MAX_FILE_SIZE_MB)

    # Word-count post extraction.
    # The 1000-word cap is enforced only for features that allow inline text input:
    # summarize, grammar_correct, translate, explain, generate_questions, generate_answers.
    extracted_word_count: Optional[int] = Field(default=None, ge=0)

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
    # For text-based AI processing actions, extracted text is required.
    # For conversion, redaction, data masking, structured extraction, and compliance,
    # text can be omitted when the backend derives it from the uploaded file.
    text: Optional[NonEmptyStr] = None
    metadata: DocumentMetadata

    # Real persisted file reference (set by upload layer)
    filename: Optional[NonEmptyStr] = None
    mime_type: Optional[NonEmptyStr] = None


class DocumentSetPayload(BaseModel):
    """
    Used for workflows that may operate on a provided document set instead of a single document.
    Per contract this applies to Structured Extraction and Compliance.
    """
    documents: List[DocumentPayload] = Field(..., min_length=1)


class MediaPayload(BaseModel):
    media_type: MediaType
    media_format: Union[AudioFormat, VideoFormat]
    file_size_mb: float = Field(..., ge=0)
    duration_seconds: int = Field(..., ge=1)
    filename: Optional[NonEmptyStr] = None
    mime_type: Optional[NonEmptyStr] = None

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


InputArtifact = Union[DocumentPayload, DocumentSetPayload, MediaPayload]


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
                f"Unsupported conversion pair: {input_format.value} -> {self.output_format.value} "
                f"(strict v1 contract)."
            )


class SummarizationRequest(BaseModel):
    feature: Literal[FeatureType.summarize]


class GrammarCorrectionRequest(BaseModel):
    feature: Literal[FeatureType.grammar_correct]


class TranslationRequest(BaseModel):
    """
    Product-wide supported system languages remain English/French.
    Translation itself may accept broader BCP-47-like source/target tags,
    while the backend system_language remains English/French only.
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
    medical_record = "medical_record"
    audit_document = "audit_document"


_ALL_SENSITIVE_DATA_TYPES = [item for item in SensitiveDataType]


class RedactionRequest(BaseModel):
    feature: Literal[FeatureType.redact]
    document_type: Optional[RedactionMaskingDocumentType] = None
    target_data: List[SensitiveDataType] = Field(default_factory=lambda: list(_ALL_SENSITIVE_DATA_TYPES))
    review_exclusions: List[NonEmptyStr] = Field(
        default_factory=list,
        description="Items the user reviewed and explicitly chose not to redact before final export.",
    )


class DataMaskingRequest(BaseModel):
    feature: Literal[FeatureType.data_mask]
    document_type: Optional[RedactionMaskingDocumentType] = None
    target_data: List[SensitiveDataType] = Field(default_factory=lambda: list(_ALL_SENSITIVE_DATA_TYPES))
    review_exclusions: List[NonEmptyStr] = Field(
        default_factory=list,
        description="Items the user reviewed and explicitly chose not to mask before final export.",
    )
    masking_mode: Literal["partial"] = "partial"


class StructuredExtractionDocumentClass(str, Enum):
    form = "form"
    memo = "memo"
    invoice = "invoice"
    receipt = "receipt"
    bank_statement = "bank_statement"
    kyc_document = "kyc_document"
    id_document = "id_document"
    contract = "contract"
    legal_record = "legal_record"
    medical_record = "medical_record"
    procurement_document = "procurement_document"
    technical_report = "technical_report"
    incident_report = "incident_report"
    insurance_document = "insurance_document"
    hr_record = "hr_record"
    onboarding_document = "onboarding_document"
    ticket = "ticket"


class StructuredExtractionResultShape(str, Enum):
    key_value_fields = "key_value_fields"
    tables = "tables"
    row_based_records = "row_based_records"
    machine_readable = "machine_readable"


class StructuredExtractionRequest(BaseModel):
    feature: Literal[FeatureType.structured_extract]
    document_classes: List[StructuredExtractionDocumentClass] = Field(default_factory=list)
    selected_fields: List[NonEmptyStr] = Field(
        default_factory=list,
        description="Predefined or user-selected fields to extract strictly from the provided document or document set.",
    )
    output_format: StructuredDataOutputFormat = StructuredDataOutputFormat.json
    result_shape: StructuredExtractionResultShape = StructuredExtractionResultShape.machine_readable
    allow_external_knowledge: Literal[False] = False
    require_human_review: Literal[True] = True


class ComplianceJurisdiction(str, Enum):
    nigeria = "nigeria"


class ComplianceSectorPack(str, Enum):
    nigeria_core_control_library = "nigeria_core_control_library"
    banking_and_fintech = "banking_and_fintech"
    payment_platforms_and_services = "payment_platforms_and_services"
    accounting = "accounting"
    legal_and_law = "legal_and_law"
    health = "health"
    media = "media"
    tech = "tech"
    telecom = "telecom"
    oil_and_gas = "oil_and_gas"
    energy_and_power = "energy_and_power"
    manufacturing = "manufacturing"
    pharmaceuticals = "pharmaceuticals"
    sports = "sports"
    mining = "mining"
    agriculture = "agriculture"
    maritime = "maritime"
    insurance = "insurance"
    ngo = "ngo"
    aviation = "aviation"


class ComplianceRegulatoryDomain(str, Enum):
    privacy = "privacy"
    cybersecurity = "cybersecurity"
    aml = "aml"
    consumer_protection = "consumer_protection"
    public_sector_access_to_information = "public_sector_access_to_information"
    licensing = "licensing"
    registration = "registration"
    sector_regulator_requirements = "sector_regulator_requirements"


class ComplianceReportVariant(str, Enum):
    human_readable_report = "human_readable_report"
    machine_readable_report = "machine_readable_report"
    annotated_source_output = "annotated_source_output"


class ComplianceRequest(BaseModel):
    feature: Literal[FeatureType.compliance]
    jurisdiction: Literal[ComplianceJurisdiction.nigeria] = ComplianceJurisdiction.nigeria
    sector_packs: List[ComplianceSectorPack] = Field(
        default_factory=lambda: [ComplianceSectorPack.nigeria_core_control_library],
        min_length=1,
    )
    regulatory_domains: List[ComplianceRegulatoryDomain] = Field(default_factory=list)
    report_variant: ComplianceReportVariant = ComplianceReportVariant.human_readable_report
    require_human_review: Literal[True] = True

    @model_validator(mode="after")
    def validate_sector_pack_selection(self):
        packs = set(self.sector_packs)
        if not packs:
            raise ValueError("At least one compliance sector pack must be supplied.")
        if (
            ComplianceSectorPack.nigeria_core_control_library not in packs
            and any(pack != ComplianceSectorPack.nigeria_core_control_library for pack in packs)
        ):
            raise ValueError(
                "When sector-specific packs are requested, nigeria_core_control_library must also be included."
            )
        return self


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
    questions: List[NonEmptyStr]

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
    StructuredExtractionRequest,
    ComplianceRequest,
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
    QuestionScalingRule(
        classification=QuestionScale.small,
        min_words=1,
        max_words=300,
        min_questions=4,
        max_questions=6,
    ),
    QuestionScalingRule(
        classification=QuestionScale.medium,
        min_words=301,
        max_words=700,
        min_questions=8,
        max_questions=10,
    ),
    QuestionScalingRule(
        classification=QuestionScale.large,
        min_words=701,
        max_words=1000,
        min_questions=12,
        max_questions=15,
    ),
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

_SINGLE_DOCUMENT_ACTIONS = {
    FeatureType.convert,
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.explain,
    FeatureType.redact,
    FeatureType.data_mask,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

_STRUCTURED_EXTRACTION_AND_COMPLIANCE_ACTIONS = {
    FeatureType.structured_extract,
    FeatureType.compliance,
}

EN_FR_ONLY_NON_TRANSLATE_ACTIONS = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.explain,
    FeatureType.redact,
    FeatureType.data_mask,
    FeatureType.structured_extract,
    FeatureType.compliance,
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

_STRUCTURED_EXTRACTION_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

_COMPLIANCE_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}


def _iter_documents(input_artifact: InputArtifact) -> List[DocumentPayload]:
    if isinstance(input_artifact, DocumentPayload):
        return [input_artifact]
    if isinstance(input_artifact, DocumentSetPayload):
        return list(input_artifact.documents)
    return []


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
        elif self.action in _STRUCTURED_EXTRACTION_AND_COMPLIANCE_ACTIONS:
            if not isinstance(self.input, (DocumentPayload, DocumentSetPayload)):
                raise ValueError(f"{self.action.value} requires DocumentPayload or DocumentSetPayload as input.")
        else:
            if not isinstance(self.input, DocumentPayload):
                raise ValueError(f"{self.action.value} requires DocumentPayload as input.")

        # reject client-supplied detected_language (server supplies detection)
        for document in _iter_documents(self.input):
            if document.metadata.detected_language is not None:
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

        if self.action in {FeatureType.redact, FeatureType.data_mask}:
            assert isinstance(self.input, DocumentPayload)
            if self.input.metadata.input_format not in _REDACTION_MASKING_INPUTS:
                raise ValueError(
                    f"{self.action.value} only supports input formats: pdf, docx, jpg, jpeg, png "
                    f"(strict contract rule)."
                )

        if self.action == FeatureType.structured_extract:
            for document in _iter_documents(self.input):
                if document.metadata.input_format not in _STRUCTURED_EXTRACTION_INPUTS:
                    raise ValueError(
                        "structured_extract only supports input formats: pdf, docx, jpg, jpeg, png "
                        "(strict contract rule)."
                    )

        if self.action == FeatureType.compliance:
            for document in _iter_documents(self.input):
                if document.metadata.input_format not in _COMPLIANCE_INPUTS:
                    raise ValueError(
                        "compliance only supports input formats: pdf, docx, jpg, jpeg, png "
                        "(strict contract rule)."
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
            FeatureType.structured_extract,
            FeatureType.compliance,
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
    algorithm_version: Optional[NonEmptyStr] = None


class HumanReviewRequirement(BaseModel):
    required: Literal[True] = True
    note: Literal["User or authorized reviewer verification is required before reliance or final export."] = (
        "User or authorized reviewer verification is required before reliance or final export."
    )


class InlineTextResult(BaseModel):
    output_format: Literal[InlineOutputFormat.txt] = InlineOutputFormat.txt
    content: NonEmptyStr
    meta: DeterminismMetadata


class BaseFileResult(BaseModel):
    filename: NonEmptyStr
    file_size_mb: float = Field(..., ge=0)
    storage_key: Optional[str] = None
    download_url: Optional[str] = None
    meta: DeterminismMetadata


class DocumentFileResult(BaseFileResult):
    output_format: DocumentFileOutputFormat


class StructuredExtractionFileResult(BaseFileResult):
    output_format: StructuredDataOutputFormat
    result_shape: StructuredExtractionResultShape
    selected_fields: List[NonEmptyStr] = Field(default_factory=list)


class ComplianceFileResult(BaseFileResult):
    output_format: ComplianceOutputFormat
    report_variant: ComplianceReportVariant


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
                f"Question count out of range for {rule.classification.value}: "
                f"expected {rule.min_questions}–{rule.max_questions}."
            )
        for i in range(1, n + 1):
            if not any(ln.startswith(f"{i}.") for ln in numbered):
                raise ValueError("Questions must be sequentially numbered starting at 1.")
        return self


class QuestionGenerationFileResult(DocumentFileResult):
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


class AnswerGenerationFileResult(DocumentFileResult):
    expected_question_count: int = Field(..., ge=1)


class EvidenceReference(BaseModel):
    source_document_index: int = Field(..., ge=0)
    page_number: Optional[int] = Field(default=None, ge=1)
    section_label: Optional[NonEmptyStr] = None
    locator_text: Optional[NonEmptyStr] = None
    excerpt: Optional[NonEmptyStr] = None


class ComplianceCheckStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    warning = "warning"
    missing = "missing"
    review_required = "review_required"


class RulePackVersion(BaseModel):
    sector_pack: ComplianceSectorPack
    version: NonEmptyStr


class ComplianceRuleResult(BaseModel):
    rule_id: NonEmptyStr
    rule_version: NonEmptyStr
    title: NonEmptyStr
    status: ComplianceCheckStatus
    summary: NonEmptyStr
    evidence_references: List[EvidenceReference] = Field(default_factory=list)


class ComplianceCounts(BaseModel):
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    warning: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)
    review_required: int = Field(default=0, ge=0)


class ComplianceMachineReadableReport(BaseModel):
    jurisdiction: Literal[ComplianceJurisdiction.nigeria] = ComplianceJurisdiction.nigeria
    sector_packs: List[ComplianceSectorPack] = Field(..., min_length=1)
    rule_pack_versions: List[RulePackVersion] = Field(default_factory=list)
    counts: ComplianceCounts
    rule_results: List[ComplianceRuleResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        actual = {
            ComplianceCheckStatus.passed: 0,
            ComplianceCheckStatus.failed: 0,
            ComplianceCheckStatus.warning: 0,
            ComplianceCheckStatus.missing: 0,
            ComplianceCheckStatus.review_required: 0,
        }
        for item in self.rule_results:
            actual[item.status] += 1

        expected = {
            ComplianceCheckStatus.passed: self.counts.passed,
            ComplianceCheckStatus.failed: self.counts.failed,
            ComplianceCheckStatus.warning: self.counts.warning,
            ComplianceCheckStatus.missing: self.counts.missing,
            ComplianceCheckStatus.review_required: self.counts.review_required,
        }
        if actual != expected:
            raise ValueError("ComplianceCounts must exactly match the statuses present in rule_results.")
        return self


AnalyzerResult = Union[
    InlineTextResult,
    DocumentFileResult,
    StructuredExtractionFileResult,
    ComplianceFileResult,
    QuestionGenerationInlineResult,
    QuestionGenerationFileResult,
    AnswerGenerationInlineResult,
    AnswerGenerationFileResult,
]


class AnalyzerResponse(BaseModel):
    action: FeatureType
    input_format: Union[DocumentInputFormat, Literal["audio"], Literal["video"], Literal["document_set"]]
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
    human_review: Optional[HumanReviewRequirement] = None

    @model_validator(mode="after")
    def validate_action_result_mapping_and_output_rules(self):
        # 1) Action-result family constraints
        if self.action == FeatureType.convert:
            if not isinstance(self.result, DocumentFileResult):
                raise ValueError("convert must return a document file result.")
        elif self.action == FeatureType.transcribe:
            if not isinstance(self.result, InlineTextResult):
                raise ValueError("transcribe must return inline txt only (strict contract rule).")
        elif self.action in {
            FeatureType.summarize,
            FeatureType.grammar_correct,
            FeatureType.translate,
            FeatureType.explain,
        }:
            if not isinstance(self.result, (InlineTextResult, DocumentFileResult)):
                raise ValueError(f"{self.action.value} must return inline txt or a document file result.")
        elif self.action in {FeatureType.redact, FeatureType.data_mask}:
            if not isinstance(self.result, DocumentFileResult):
                raise ValueError(f"{self.action.value} must return a document file result.")
        elif self.action == FeatureType.structured_extract:
            if not isinstance(self.result, StructuredExtractionFileResult):
                raise ValueError("structured_extract must return a structured-data file result.")
        elif self.action == FeatureType.compliance:
            if not isinstance(self.result, ComplianceFileResult):
                raise ValueError("compliance must return a compliance report file result.")
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
                    if not isinstance(self.result, DocumentFileResult):
                        raise ValueError(
                            "For pdf/docx input, output must be a downloadable file (strict contract rule)."
                        )
                    expected = (
                        DocumentFileOutputFormat.pdf
                        if self.input_format == DocumentInputFormat.pdf
                        else DocumentFileOutputFormat.docx
                    )
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
            if not isinstance(self.result, DocumentFileResult):
                raise ValueError(f"{self.action.value} output must be a downloadable file.")
            expected_map = {
                DocumentInputFormat.pdf: DocumentFileOutputFormat.pdf,
                DocumentInputFormat.docx: DocumentFileOutputFormat.docx,
                DocumentInputFormat.jpg: DocumentFileOutputFormat.jpg,
                DocumentInputFormat.jpeg: DocumentFileOutputFormat.jpeg,
                DocumentInputFormat.png: DocumentFileOutputFormat.png,
            }
            expected = expected_map[self.input_format]
            if self.result.output_format != expected:
                raise ValueError("Output file extension must match input extension (strict contract rule).")

        if self.action == FeatureType.structured_extract:
            if not isinstance(self.result, StructuredExtractionFileResult):
                raise ValueError("structured_extract output must be a structured-data file result.")

        if self.action == FeatureType.compliance:
            if not isinstance(self.result, ComplianceFileResult):
                raise ValueError("compliance output must be a compliance file result.")
            if self.result.report_variant == ComplianceReportVariant.machine_readable_report:
                if self.result.output_format != ComplianceOutputFormat.json:
                    raise ValueError("machine_readable_report must use json output.")
            else:
                if self.result.output_format != ComplianceOutputFormat.pdf:
                    raise ValueError("human_readable_report and annotated_source_output must use pdf output.")

        # 3) Transcription output rule: always inline txt
        if self.action == FeatureType.transcribe:
            if not isinstance(self.result, InlineTextResult) or self.result.output_format != InlineOutputFormat.txt:
                raise ValueError("transcribe output must be inline txt (strict contract rule).")

        # 4) Language boundary validation for non-translate features (when language fields are provided)
        if self.action in EN_FR_ONLY_NON_TRANSLATE_ACTIONS or self.action == FeatureType.transcribe:
            if self.detected_language is not None and not _is_en_or_fr_tag(self.detected_language):
                raise ValueError("Non-translate responses must have detected_language en*/fr* when provided.")
            if self.output_language is not None and not _is_en_or_fr_tag(self.output_language):
                raise ValueError("Non-translate responses must have output_language en*/fr* when provided.")
            if self.detected_language and self.output_language:
                d = self.detected_language.lower().replace("_", "-")
                o = self.output_language.lower().replace("_", "-")
                if (d.startswith("en") and not o.startswith("en")) or (d.startswith("fr") and not o.startswith("fr")):
                    raise ValueError(
                        "For non-translate features, output_language must match detected_language "
                        "(keep same language)."
                    )

        # 5) Optional consistency check between selected system_language and response language
        if self.action in EN_FR_ONLY_NON_TRANSLATE_ACTIONS or self.action == FeatureType.transcribe:
            forced_tag = _system_language_to_tag(self.system_language)
            if self.output_language is not None:
                o = self.output_language.lower().replace("_", "-")
                if forced_tag == "en" and not (o == "en" or o.startswith("en-")):
                    raise ValueError("system_language='english' but output_language is not en*.")
                if forced_tag == "fr" and not (o == "fr" or o.startswith("fr-")):
                    raise ValueError("system_language='french' but output_language is not fr*.")

        # 6) Human review enforcement
        if self.action in {
            FeatureType.redact,
            FeatureType.data_mask,
            FeatureType.structured_extract,
            FeatureType.compliance,
        }:
            if self.human_review is None or self.human_review.required is not True:
                raise ValueError(
                    f"{self.action.value} responses must explicitly declare human review as required."
                )

        return self

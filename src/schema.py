from enum import Enum
from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field, StringConstraints, field_validator

# CONTRACT CONSTANTS (HARD LIMITS)

MAX_FILE_SIZE_MB = 10
MAX_WORD_COUNT = 1000
MIN_WORD_COUNT = 1

# SUPPORTED FORMATS (STRICTLY V1)

class InputFormat(str, Enum):
    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    jpg = "jpg"
    jpeg = "jpeg"
class OutputFormat(str, Enum):
    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    structured_text = "structured_text"

# FEATURE ENUM (ONLY V1 FEATURES)

class FeatureType(str, Enum):
    convert = "convert"
    summarize = "summarize"
    grammar_correct = "grammar_correct"
    translate = "translate"
    explain = "explain"
    generate_questions = "generate_questions"
    generate_answers = "generate_answers"

# DOCUMENT METADATA

class DocumentMetadata(BaseModel):
    input_format: InputFormat
    file_size_mb: float = Field(..., ge=0, le=MAX_FILE_SIZE_MB)
    extracted_word_count: int = Field(..., ge= MIN_WORD_COUNT, le=MAX_WORD_COUNT)
    ocr_used: bool
    @field_validator("ocr_used")
    @classmethod
    def validate_ocr_usage(cls, v, info):
        fmt = info.data.get("input_format")
        if fmt in (InputFormat.jpg, InputFormat.jpeg) and not v:
            raise ValueError("OCR must be enabled for image inputs")
        if fmt not in (InputFormat.jpg, InputFormat.jpeg) and v:
            raise ValueError("OCR is allowed only for image inputs")
        return v

# DOCUMENT PAYLOAD (STATELESS CONTENT WRAPPER)

class DocumentPayload(BaseModel):
    text: Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1)]
    metadata: DocumentMetadata

# FEATURE-SPECIFIC REQUEST PAYLOADS

class ConversionRequest(BaseModel):
    feature: Literal[FeatureType.convert]
    output_format: OutputFormat
class SummarizationRequest(BaseModel):
    feature: Literal[FeatureType.summarize]
class GrammarCorrectionRequest(BaseModel):
    feature: Literal[FeatureType.grammar_correct]
class TranslationRequest(BaseModel):
    feature: Literal[FeatureType.translate]
    target_language: Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1)]
class ExplanationRequest(BaseModel):
    feature: Literal[FeatureType.explain]
class QuestionGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_questions]
class AnswerGenerationRequest(BaseModel):
    feature: Literal[FeatureType.generate_answers]
    questions: List[Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)]]
    @field_validator("questions")
    @classmethod
    def validate_numbered_questions(cls, v):
        if not v:
            raise ValueError("Questions list cannot be empty")
        for i, q in enumerate(v, start=1):
            if not q.lstrip().startswith(f"{i}."):
                raise ValueError("Questions must be sequentially numbered starting at 1")
        return v

# UNIFIED REQUEST ENVELOPE (STATELESS)

class AnalyzerRequest(BaseModel):
    action: FeatureType
    document: DocumentPayload
    payload: Optional[ConversionRequest|SummarizationRequest|GrammarCorrectionRequest|TranslationRequest
|ExplanationRequest|QuestionGenerationRequest|AnswerGenerationRequest]

# QUESTION SCALING CONTRACT

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
    @field_validator("max_words")
    @classmethod
    def validate_word_bounds(cls, v, info):
        min_words = info.data.get("min_words")
        if min_words is not None and v <= min_words:
            raise ValueError("Invalid word range")
        return v
    @field_validator("max_questions")
    @classmethod
    def validate_question_bounds(cls, v, info):
        min_questions = info.data.get("min_questions")
        if min_questions is not None and v <= min_questions:
            raise ValueError("Invalid question range")
        return v
QUESTION_SCALING_RULES = [
    QuestionScalingRule(
        classification=QuestionScale.small,
        min_words=1,
        max_words=300,
        min_questions=4,
        max_questions=6
    ),
    QuestionScalingRule(
        classification=QuestionScale.medium,
        min_words=301,
        max_words=700,
        min_questions=8,
        max_questions=10
    ),
    QuestionScalingRule(
        classification=QuestionScale.large,
        min_words=701,
        max_words=1000,
        min_questions=12,
        max_questions=15
    ),
]

# DETERMINISTIC RESPONSE STRUCTURES

class StructuredTextResponse(BaseModel):
    content: Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1)]
class NumberedListResponse(BaseModel):
    items: List[Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)]]
    @field_validator("items")
    @classmethod
    def validate_sequential_numbering(cls, v):
        if not v:
            raise ValueError("List cannot be empty")
        for i, item in enumerate(v, start=1):
            if not item.lstrip().startswith(f"{i}."):
                raise ValueError("Items must be sequentially numbered starting at 1")
        return v

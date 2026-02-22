from typing import Type
from src.schema import (
    AnalyzerRequest,
    FeatureType,
    QUESTION_SCALING_RULES,
    QuestionScale,
    StructuredTextResponse,
    NumberedListResponse,
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    ExplanationRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
)

# CORE VALIDATION FUNCTIONS

def validate_action_payload_consistency(request: AnalyzerRequest) -> None:
    """
    Ensures that the payload matches the declared action.
    """
    if request.payload is None:
        raise ValueError("Payload is required")
    action_payload_map: dict[FeatureType, Type] = {
        FeatureType.convert: ConversionRequest,
        FeatureType.summarize: SummarizationRequest,
        FeatureType.grammar_correct: GrammarCorrectionRequest,
        FeatureType.translate: TranslationRequest,
        FeatureType.explain: ExplanationRequest,
        FeatureType.generate_questions: QuestionGenerationRequest,
        FeatureType.generate_answers: AnswerGenerationRequest,
    }
    expected_model = action_payload_map.get(request.action)
    if not isinstance(request.payload, expected_model):
        raise ValueError("Payload does not match declared action")

def validate_word_count_bounds(request: AnalyzerRequest) -> None:
    """
    Ensures document word count remains inside contract limits.
    """
    word_count = request.document.metadata.extracted_word_count
    if word_count < 1:
        raise ValueError("Document contains no words")
    if word_count > 1000:
        raise ValueError("Document exceeds maximum allowed word count")

# QUESTION SCALING LOGIC

def classify_question_scale(word_count: int) -> QuestionScale:
    """
    Determines the question scale classification
    based on deterministic scaling rules.
    """
    for rule in QUESTION_SCALING_RULES:
        if rule.min_words <= word_count <= rule.max_words:
            return rule.classification
    raise ValueError("Word count outside supported scaling rules")

def get_question_range(word_count: int) -> tuple[int, int]:
    """
    Returns (min_questions, max_questions)
    based on document word count.
    """
    for rule in QUESTION_SCALING_RULES:
        if rule.min_words <= word_count <= rule.max_words:
            return rule.min_questions, rule.max_questions
    raise ValueError("Word count outside supported scaling rules")

# RESPONSE VALIDATORS

def validate_structured_text_response(content: str) -> StructuredTextResponse:
    """
    Ensures structured text response meets schema contract.
    """
    if not content or not content.strip():
        raise ValueError("Response content cannot be empty")
    return StructuredTextResponse(content=content)

def validate_numbered_list_response(items: list[str]) -> NumberedListResponse:
    """
    Ensures numbered list response is properly sequential.
    """
    if not items:
        raise ValueError("Response list cannot be empty")

    # Pydantic model will re-check numbering deterministically
    
    return NumberedListResponse(items=items)

# MASTER REQUEST VALIDATOR

def validate_analyzer_request(
    request: AnalyzerRequest,
    usage_snapshot
) -> None:
    """
    Full deterministic validation pipeline.
    """
    validate_action_payload_consistency(request)
    validate_word_count_bounds(request)

    # Additional deterministic checks for specific features

    if request.action == FeatureType.generate_questions:
        word_count = request.document.metadata.extracted_word_count
        classify_question_scale(word_count)
    if request.action == FeatureType.generate_answers:
        if not request.payload.questions:
            raise ValueError("Questions list cannot be empty")

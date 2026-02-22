import pytest
from src.validation import (
    validate_action_payload_consistency,
    validate_word_count_bounds,
    classify_question_scale,
    get_question_range,
    validate_structured_text_response,
    validate_numbered_list_response,
    validate_analyzer_request
)
from src.schema import (
    AnalyzerRequest,
    FeatureType,
    ConversionRequest,
    SummarizationRequest,
    GrammarCorrectionRequest,
    TranslationRequest,
    ExplanationRequest,
    QuestionGenerationRequest,
    AnswerGenerationRequest,
    QuestionScale,
    StructuredTextResponse,
    NumberedListResponse,
    QUESTION_SCALING_RULES,
    OutputFormat,
    DocumentPayload,
    DocumentMetadata
)

# Helper: create DocumentPayload

def make_document(word_count: int, text: str = "Sample text content") -> DocumentPayload:
    # Ensure Pydantic-valid value (1-1000)
    valid_wc = max(1, min(word_count, 1000))
    return DocumentPayload(
        text=text,
        metadata=DocumentMetadata(
            input_format="txt",
            file_size_mb=0.1,
            extracted_word_count=valid_wc,
            ocr_used=False
        )
    )

# Payload Mocks

mock_payloads = {
    FeatureType.convert: ConversionRequest,
    FeatureType.summarize: SummarizationRequest,
    FeatureType.grammar_correct: GrammarCorrectionRequest,
    FeatureType.translate: TranslationRequest,
    FeatureType.explain: ExplanationRequest,
    FeatureType.generate_questions: QuestionGenerationRequest,
    FeatureType.generate_answers: AnswerGenerationRequest,
}

# TEST: validate_action_payload_consistency

@pytest.mark.parametrize("feature_type,payload_class", mock_payloads.items())
def test_validate_action_payload_consistency_valid(feature_type, payload_class):
    if payload_class is ConversionRequest:
        payload_instance = payload_class(feature=FeatureType.convert, output_format=OutputFormat.txt)
    elif payload_class is TranslationRequest:
        payload_instance = payload_class(feature=FeatureType.translate, target_language="fr")
    elif payload_class is AnswerGenerationRequest:
        payload_instance = payload_class(
            feature=FeatureType.generate_answers,
            questions=["1. What is this?", "2. How does it work?"]
        )
    elif payload_class is QuestionGenerationRequest:
        payload_instance = payload_class(feature=FeatureType.generate_questions)
    else:
        payload_instance = payload_class(feature=feature_type)

    request = AnalyzerRequest(action=feature_type, payload=payload_instance, document=make_document(10))
    validate_action_payload_consistency(request)

def test_validate_action_payload_consistency_invalid_payload():
    request = AnalyzerRequest(
        action=FeatureType.convert,
        payload=SummarizationRequest(feature=FeatureType.summarize),
        document=make_document(10)
    )
    with pytest.raises(ValueError, match="Payload does not match declared action"):
        validate_action_payload_consistency(request)

def test_validate_action_payload_consistency_missing_payload():
    request = AnalyzerRequest(action=FeatureType.convert, payload=None, document=make_document(10))
    with pytest.raises(ValueError, match="Payload is required"):
        validate_action_payload_consistency(request)

# TEST: validate_word_count_bounds

def test_validate_word_count_bounds_valid():
    request = AnalyzerRequest(
        action=FeatureType.summarize,
        payload=SummarizationRequest(feature=FeatureType.summarize),
        document=make_document(500)
    )
    validate_word_count_bounds(request)

def test_validate_word_count_bounds_too_small():
    doc = make_document(1)  # Pydantic-valid
    request = AnalyzerRequest(
        action=FeatureType.summarize,
        payload=SummarizationRequest(feature=FeatureType.summarize),
        document=doc
    )
    
    # override to simulate invalid value
    
    request.document.metadata.extracted_word_count = 0
    with pytest.raises(ValueError, match="Document contains no words"):
        validate_word_count_bounds(request)

def test_validate_word_count_bounds_too_large():
    doc = make_document(1000)  # Pydantic-valid
    request = AnalyzerRequest(
        action=FeatureType.summarize,
        payload=SummarizationRequest(feature=FeatureType.summarize),
        document=doc
    )
    request.document.metadata.extracted_word_count = 1001
    with pytest.raises(ValueError, match="Document exceeds maximum allowed word count"):
        validate_word_count_bounds(request)

# TEST: classify_question_scale & get_question_range

def test_classify_question_scale_valid():
    for rule in QUESTION_SCALING_RULES:
        assert classify_question_scale(rule.min_words) == rule.classification
        assert classify_question_scale(rule.max_words) == rule.classification

def test_classify_question_scale_invalid():
    with pytest.raises(ValueError, match="Word count outside supported scaling rules"):
        classify_question_scale(-1)

def test_get_question_range_valid():
    for rule in QUESTION_SCALING_RULES:
        min_q, max_q = get_question_range(rule.min_words)
        assert min_q == rule.min_questions
        assert max_q == rule.max_questions

def test_get_question_range_invalid():
    with pytest.raises(ValueError, match="Word count outside supported scaling rules"):
        get_question_range(-5)

# TEST: validate_structured_text_response

def test_validate_structured_text_response_valid():
    result = validate_structured_text_response("Hello world")
    assert isinstance(result, StructuredTextResponse)
    assert result.content == "Hello world"

@pytest.mark.parametrize("content", ["", "   ", None])
def test_validate_structured_text_response_invalid(content):
    with pytest.raises(ValueError, match="Response content cannot be empty"):
        validate_structured_text_response(content)

# TEST: validate_numbered_list_response

def test_validate_numbered_list_response_valid():
    items = ["1. One", "2. Two", "3. Three"]
    result = validate_numbered_list_response(items)
    assert isinstance(result, NumberedListResponse)
    assert result.items == items

def test_validate_numbered_list_response_invalid():
    with pytest.raises(ValueError, match="Response list cannot be empty"):
        validate_numbered_list_response([])

# TEST: validate_analyzer_request

def test_validate_analyzer_request_generate_questions():
    payload = QuestionGenerationRequest(feature=FeatureType.generate_questions)
    doc = make_document(50)
    request = AnalyzerRequest(action=FeatureType.generate_questions, payload=payload, document=doc)
    validate_analyzer_request(request, usage_snapshot=None)

def test_validate_analyzer_request_generate_answers_missing_questions():
   
    # Initialize with Pydantic-valid sequential questions
    
    payload = AnswerGenerationRequest(
        feature=FeatureType.generate_answers,
        questions=["1. What is this?", "2. How does it work?"]
    )
    doc = make_document(50)
    request = AnalyzerRequest(action=FeatureType.generate_answers, payload=payload, document=doc)
    
    # override to empty list to trigger validation
    
    request.payload.questions = []
    with pytest.raises(ValueError, match="Questions list cannot be empty"):
        validate_analyzer_request(request, usage_snapshot=None)
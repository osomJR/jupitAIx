from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import docx  # python-docx
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

from .schema import (
    DocumentInputFormat,
    DocumentMetadata,
    DocumentPayload,
    DocumentSetPayload,
    FeatureType,
    MAX_FILE_SIZE_MB,
    TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT,
    classify_word_count,
)

# OCR configuration
OCR_CONFIG = "--oem 3 --psm 6"

# Common BCP-47 / language-name aliases -> Tesseract traineddata codes.
# This is only used when callers supply explicit OCR language hints.
_TESSERACT_LANG_ALIASES: dict[str, str] = {
    "ar": "ara",
    "arabic": "ara",
    "de": "deu",
    "german": "deu",
    "en": "eng",
    "english": "eng",
    "es": "spa",
    "spanish": "spa",
    "fa": "fas",
    "farsi": "fas",
    "fr": "fra",
    "french": "fra",
    "ha": "hau",
    "hausa": "hau",
    "hi": "hin",
    "hindi": "hin",
    "ig": "ibo",
    "igbo": "ibo",
    "it": "ita",
    "italian": "ita",
    "ja": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "korean": "kor",
    "nl": "nld",
    "dutch": "nld",
    "pl": "pol",
    "polish": "pol",
    "pt": "por",
    "pt-br": "por",
    "portuguese": "por",
    "ru": "rus",
    "russian": "rus",
    "sw": "swa",
    "swahili": "swa",
    "tr": "tur",
    "turkish": "tur",
    "uk": "ukr",
    "ukrainian": "ukr",
    "yo": "yor",
    "yoruba": "yor",
    "zh": "chi_sim",
    "zh-cn": "chi_sim",
    "zh-hans": "chi_sim",
    "zh-tw": "chi_tra",
    "zh-hant": "chi_tra",
}

CONVERSION_ACTIONS = {FeatureType.convert}

TEXT_AI_DOC_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.txt,
}

REDACTION_MASKING_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

STRUCTURED_EXTRACTION_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

COMPLIANCE_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

CONVERSION_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}

OPTIONAL_TEXT_DOCUMENT_ACTIONS = {
    FeatureType.redact,
    FeatureType.data_mask,
    FeatureType.structured_extract,
    FeatureType.compliance,
}

DOCUMENT_SET_ACTIONS = {
    FeatureType.structured_extract,
    FeatureType.compliance,
}

_ALLOWED_INPUT_FORMATS_BY_ACTION: dict[FeatureType, set[DocumentInputFormat]] = {
    FeatureType.convert: CONVERSION_INPUT_FORMATS,
    FeatureType.summarize: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.grammar_correct: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.translate: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.explain: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.generate_questions: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.generate_answers: TEXT_AI_DOC_INPUT_FORMATS,
    FeatureType.redact: REDACTION_MASKING_INPUT_FORMATS,
    FeatureType.data_mask: REDACTION_MASKING_INPUT_FORMATS,
    FeatureType.structured_extract: STRUCTURED_EXTRACTION_INPUT_FORMATS,
    FeatureType.compliance: COMPLIANCE_INPUT_FORMATS,
}


# ----------------------------
# OCR helpers
# ----------------------------
def get_available_tesseract_languages() -> list[str]:
    """Return installed Tesseract language codes, excluding utility packs."""
    languages = pytesseract.get_languages(config="")
    return sorted(lang for lang in languages if lang not in {"osd", "equ"})



def _normalize_ocr_language_token(token: str) -> str:
    normalized = token.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("OCR language token cannot be empty.")
    return _TESSERACT_LANG_ALIASES.get(normalized, normalized)



def resolve_ocr_lang(ocr_languages: Optional[Sequence[str]] = None) -> str:
    """
    Resolve the Tesseract language bundle.

    Behavior:
    - if caller supplies OCR language hints, normalize and validate them
    - otherwise, use all installed OCR languages to maximize scanned-document coverage
    - include osd when present for orientation/script detection support
    """
    available = set(get_available_tesseract_languages())
    raw_available = set(pytesseract.get_languages(config=""))

    if ocr_languages:
        requested: list[str] = []
        for item in ocr_languages:
            code = _normalize_ocr_language_token(item)
            if code not in available:
                raise ValueError(
                    f"Requested OCR language '{item}' resolved to '{code}', "
                    "but that traineddata is not installed in Tesseract."
                )
            if code not in requested:
                requested.append(code)
        selected = requested
    else:
        selected = sorted(available)

    if not selected:
        raise ValueError(
            "No OCR languages are installed in Tesseract. Install traineddata files "
            "for the languages you need before processing scanned documents."
        )

    if "osd" in raw_available:
        selected = [*selected, "osd"]

    return "+".join(selected)


# ----------------------------
# File helpers
# ----------------------------
def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy by cleaning the image.
    Steps:
    - convert to grayscale
    - denoise
    - adaptive threshold
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    return Image.fromarray(thresh)



def get_file_size_mb(file_path: Path) -> float:
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb <= 0:
        raise ValueError("File is empty.")
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB.")

    return round(size_mb, 4)



def detect_format(file_path: Path) -> DocumentInputFormat:
    suffix = file_path.suffix.lower().lstrip(".")
    try:
        return DocumentInputFormat(suffix)
    except ValueError as exc:
        raise ValueError(f"Unsupported file format: {suffix}") from exc


# ----------------------------
# Text extraction
# ----------------------------
def extract_text_from_txt(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("TXT file must be valid UTF-8.") from exc



def extract_text_from_docx(file_path: Path) -> str:
    document = docx.Document(file_path)
    return "\n".join(p.text for p in document.paragraphs).strip()



def extract_text_from_image(file_path: Path, *, ocr_lang: str) -> str:
    image = Image.open(file_path).convert("RGB")
    processed = preprocess_for_ocr(image)
    return pytesseract.image_to_string(processed, lang=ocr_lang, config=OCR_CONFIG).strip()



def extract_text_from_pdf_text(file_path: Path) -> str:
    chunks: list[str] = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            chunks.append(page.get_text())
    return "\n".join(chunks).strip()



def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)



def extract_text_from_pdf_ocr(file_path: Path, *, ocr_lang: str, zoom: float = 4.0) -> str:
    matrix = fitz.Matrix(zoom, zoom)
    chunks: list[str] = []

    with fitz.open(file_path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=matrix)
            image = _pixmap_to_pil(pix).convert("RGB")
            processed = preprocess_for_ocr(image)
            chunks.append(pytesseract.image_to_string(processed, lang=ocr_lang, config=OCR_CONFIG))

    return "\n".join(chunks).strip()



def extract_text_by_format(
    file_path: Path,
    fmt: DocumentInputFormat,
    *,
    ocr_languages: Optional[Sequence[str]] = None,
) -> tuple[str, bool]:
    """
    Return (text, ocr_used).

    Contract alignment:
    - txt/docx => ocr_used=False
    - pdf => OCR fallback only if native extraction is empty
    - images => OCR always used when text extraction is attempted
    """
    if fmt == DocumentInputFormat.txt:
        return extract_text_from_txt(file_path), False

    if fmt == DocumentInputFormat.docx:
        return extract_text_from_docx(file_path), False

    ocr_lang = resolve_ocr_lang(ocr_languages)

    if fmt == DocumentInputFormat.pdf:
        text = extract_text_from_pdf_text(file_path)
        if text:
            return text, False
        return extract_text_from_pdf_ocr(file_path, ocr_lang=ocr_lang), True

    if fmt in (DocumentInputFormat.jpg, DocumentInputFormat.jpeg, DocumentInputFormat.png):
        return extract_text_from_image(file_path, ocr_lang=ocr_lang), True

    raise ValueError(f"Unsupported file format: {fmt.value}")


# ----------------------------
# Word-count helpers
# ----------------------------
def count_words(text: str) -> int:
    return len(text.split())



def enforce_text_ai_word_contract(word_count: int) -> None:
    if word_count < 1:
        raise ValueError("Document contains no words.")
    classify_word_count(word_count)


# ----------------------------
# Payload builders
# ----------------------------
def build_inline_text_payload(
    text: str,
    *,
    input_format: DocumentInputFormat = DocumentInputFormat.txt,
) -> DocumentPayload:
    """
    Normalize inline text into a schema-compliant DocumentPayload.

    Notes:
    - Inline text is represented as txt input.
    - detected_language is intentionally omitted because request-side validation
      forbids client-supplied detected_language; analyzer/server handles detection.
    - The strict 1..1000 word-count contract is enforced only for text AI actions.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("Inline text cannot be empty.")

    if input_format != DocumentInputFormat.txt:
        raise ValueError("Inline text payload must use input_format='txt'.")

    word_count = count_words(normalized)
    enforce_text_ai_word_contract(word_count)

    metadata = DocumentMetadata(
        input_format=DocumentInputFormat.txt,
        file_size_mb=0.0,
        extracted_word_count=word_count,
        ocr_used=False,
    )
    return DocumentPayload(text=normalized, metadata=metadata)



def _build_document_payload(
    file_path: str,
    *,
    allowed_formats: set[DocumentInputFormat],
    require_text: bool,
    enforce_text_ai_range: bool,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentPayload:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("File not found.")

    fmt = detect_format(path)
    if fmt not in allowed_formats:
        allowed = ", ".join(item.value for item in sorted(allowed_formats, key=lambda x: x.value))
        raise ValueError(f"Unsupported input format for this action. Allowed formats: {allowed}.")

    file_size_mb = get_file_size_mb(path)

    text: Optional[str] = None
    extracted_word_count: Optional[int] = None
    ocr_used = False

    # Conversion does not require extracted text or word count.
    if fmt != DocumentInputFormat.txt or require_text or enforce_text_ai_range:
        if fmt in {
            DocumentInputFormat.txt,
            DocumentInputFormat.docx,
            DocumentInputFormat.pdf,
            DocumentInputFormat.jpg,
            DocumentInputFormat.jpeg,
            DocumentInputFormat.png,
        }:
            extracted_text, ocr_used = extract_text_by_format(path, fmt, ocr_languages=ocr_languages)
            normalized = extracted_text.strip()
            if normalized:
                text = normalized
                extracted_word_count = count_words(normalized)

    if require_text and not text:
        raise ValueError("Document text could not be extracted.")

    if text and enforce_text_ai_range:
        enforce_text_ai_word_contract(extracted_word_count or 0)

    metadata = DocumentMetadata(
        input_format=fmt,
        file_size_mb=file_size_mb,
        extracted_word_count=extracted_word_count,
        ocr_used=ocr_used,
    )
    return DocumentPayload(text=text, metadata=metadata, filename=path.name)



def build_conversion_document_payload(file_path: str) -> DocumentPayload:
    """
    Build a DocumentPayload for the convert action.

    Contract alignment:
    - convert accepts pdf/docx/jpg/jpeg/png
    - text is optional
    - extracted_word_count is optional
    """
    return _build_document_payload(
        file_path,
        allowed_formats=CONVERSION_INPUT_FORMATS,
        require_text=False,
        enforce_text_ai_range=False,
    )



def build_text_ai_document_payload(
    file_path: str,
    *,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentPayload:
    """
    Build a DocumentPayload for text AI document actions:
    summarize, grammar_correct, translate, explain, generate_questions, generate_answers.

    Contract alignment:
    - only pdf/docx/txt are allowed
    - extracted text is required
    - extracted_word_count is required and must fit the 1..1000 contract range
    """
    return _build_document_payload(
        file_path,
        allowed_formats=TEXT_AI_DOC_INPUT_FORMATS,
        require_text=True,
        enforce_text_ai_range=True,
        ocr_languages=ocr_languages,
    )



def build_redaction_or_masking_document_payload(
    file_path: str,
    *,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentPayload:
    """
    Build a DocumentPayload for redact/data_mask.

    Contract alignment:
    - accepts pdf/docx/jpg/jpeg/png
    - text is optional in schema
    - extracted_word_count may be present, but is not capped at 1000 here
    """
    return _build_document_payload(
        file_path,
        allowed_formats=REDACTION_MASKING_INPUT_FORMATS,
        require_text=False,
        enforce_text_ai_range=False,
        ocr_languages=ocr_languages,
    )



def build_structured_extraction_or_compliance_document_payload(
    file_path: str,
    *,
    action: FeatureType,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentPayload:
    """
    Build a single-document payload for structured_extract or compliance.

    Contract alignment:
    - accepts pdf/docx/jpg/jpeg/png
    - text is optional in schema
    - extracted_word_count may be present, but is not capped at 1000 here
    """
    if action not in DOCUMENT_SET_ACTIONS:
        raise ValueError("This builder only supports structured_extract and compliance actions.")

    allowed_formats = _ALLOWED_INPUT_FORMATS_BY_ACTION[action]
    return _build_document_payload(
        file_path,
        allowed_formats=allowed_formats,
        require_text=False,
        enforce_text_ai_range=False,
        ocr_languages=ocr_languages,
    )



def build_document_set_payload(
    file_paths: Sequence[str],
    *,
    action: FeatureType,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentSetPayload:
    """
    Build a DocumentSetPayload for structured_extract or compliance.
    """
    if action not in DOCUMENT_SET_ACTIONS:
        raise ValueError("Document sets are only supported for structured_extract and compliance.")
    if not file_paths:
        raise ValueError("file_paths cannot be empty.")

    documents = [
        build_structured_extraction_or_compliance_document_payload(
            file_path,
            action=action,
            ocr_languages=ocr_languages,
        )
        for file_path in file_paths
    ]
    return DocumentSetPayload(documents=documents)



def build_document_payload_for_action(
    *,
    action: FeatureType,
    file_path: Optional[str] = None,
    inline_text: Optional[str] = None,
    ocr_languages: Optional[Sequence[str]] = None,
) -> DocumentPayload:
    """
    Runtime-aware normalization entrypoint for single-document actions.

    Rules:
    - inline_text => txt payload for text AI document actions only
    - convert => text optional
    - text AI actions => extracted text + extracted_word_count required
    - redact/data_mask/structured_extract/compliance => extracted text optional
    """
    if inline_text is not None:
        if file_path is not None:
            raise ValueError("Provide either file_path or inline_text, not both.")
        if action not in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
            raise ValueError(f"{action.value} does not support inline text through this extraction path.")
        return build_inline_text_payload(inline_text)

    if file_path is None:
        raise ValueError("Either file_path or inline_text must be provided.")

    if action in CONVERSION_ACTIONS:
        return build_conversion_document_payload(file_path)

    if action in TEXT_AI_DOC_ACTIONS_REQUIRING_TEXT_AND_WORDCOUNT:
        return build_text_ai_document_payload(file_path, ocr_languages=ocr_languages)

    if action in {FeatureType.redact, FeatureType.data_mask}:
        return build_redaction_or_masking_document_payload(file_path, ocr_languages=ocr_languages)

    if action in DOCUMENT_SET_ACTIONS:
        return build_structured_extraction_or_compliance_document_payload(
            file_path,
            action=action,
            ocr_languages=ocr_languages,
        )

    raise ValueError(f"Unsupported document action for extraction: {action.value}")



def build_input_artifact_for_action(
    *,
    action: FeatureType,
    file_path: Optional[str] = None,
    file_paths: Optional[Sequence[str]] = None,
    inline_text: Optional[str] = None,
    ocr_languages: Optional[Sequence[str]] = None,
) -> Union[DocumentPayload, DocumentSetPayload]:
    """
    Runtime-aware normalization entrypoint aligned with schema InputArtifact rules.

    - Most document actions return DocumentPayload.
    - structured_extract and compliance may return DocumentSetPayload when file_paths is supplied.
    - inline_text is supported only for text AI document actions and produces txt DocumentPayload.
    """
    provided = sum(
        value is not None
        for value in (
            file_path,
            file_paths,
            inline_text,
        )
    )
    if provided != 1:
        raise ValueError("Provide exactly one of file_path, file_paths, or inline_text.")

    if inline_text is not None:
        return build_document_payload_for_action(action=action, inline_text=inline_text)

    if file_paths is not None:
        return build_document_set_payload(file_paths, action=action, ocr_languages=ocr_languages)

    return build_document_payload_for_action(
        action=action,
        file_path=file_path,
        ocr_languages=ocr_languages,
    )


__all__ = [
    "OCR_CONFIG",
    "CONVERSION_ACTIONS",
    "TEXT_AI_DOC_INPUT_FORMATS",
    "REDACTION_MASKING_INPUT_FORMATS",
    "STRUCTURED_EXTRACTION_INPUT_FORMATS",
    "COMPLIANCE_INPUT_FORMATS",
    "CONVERSION_INPUT_FORMATS",
    "OPTIONAL_TEXT_DOCUMENT_ACTIONS",
    "DOCUMENT_SET_ACTIONS",
    "get_available_tesseract_languages",
    "resolve_ocr_lang",
    "preprocess_for_ocr",
    "get_file_size_mb",
    "detect_format",
    "extract_text_from_txt",
    "extract_text_from_docx",
    "extract_text_from_image",
    "extract_text_from_pdf_text",
    "extract_text_from_pdf_ocr",
    "extract_text_by_format",
    "count_words",
    "enforce_text_ai_word_contract",
    "build_inline_text_payload",
    "build_conversion_document_payload",
    "build_text_ai_document_payload",
    "build_redaction_or_masking_document_payload",
    "build_structured_extraction_or_compliance_document_payload",
    "build_document_set_payload",
    "build_document_payload_for_action",
    "build_input_artifact_for_action",
]

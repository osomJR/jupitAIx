from __future__ import annotations


from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import fitz  # PyMuPDF
import docx  # python-docx
import pytesseract
from PIL import Image

from .schema import (
    DocumentInputFormat,
    DocumentMetadata,
    DocumentPayload,
    FeatureType,
    MAX_FILE_SIZE_MB,
    MAX_WORD_COUNT,
)

OCR_LANG = "eng+fra+osd"
OCR_CONFIG = "--oem 3 --psm 6"

AI_DOC_ACTIONS = {
    FeatureType.summarize,
    FeatureType.grammar_correct,
    FeatureType.translate,
    FeatureType.explain,
    FeatureType.generate_questions,
    FeatureType.generate_answers,
}

CONVERSION_ACTIONS = {
    FeatureType.convert,
}

AI_DOC_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.txt,
}

CONVERSION_INPUT_FORMATS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}


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

    # convert PIL → OpenCV
    img = np.array(image)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # noise reduction
    gray = cv2.medianBlur(gray, 3)

    # adaptive threshold (handles uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # return back to PIL
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


def extract_text_from_image(file_path: Path) -> str:
    image = Image.open(file_path).convert("RGB")
    processed = preprocess_for_ocr(image)
    return pytesseract.image_to_string(processed, lang=OCR_LANG, config=OCR_CONFIG).strip()


def extract_text_from_pdf_text(file_path: Path) -> str:
    chunks: list[str] = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            chunks.append(page.get_text())
    return "\n".join(chunks).strip()


def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def extract_text_from_pdf_ocr(file_path: Path, *, zoom: float = 4.0) -> str:
    matrix = fitz.Matrix(zoom, zoom)
    chunks: list[str] = []

    with fitz.open(file_path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=matrix)
            image = _pixmap_to_pil(pix).convert("RGB")
            processed = preprocess_for_ocr(image)
            chunks.append(pytesseract.image_to_string(processed, lang=OCR_LANG, config=OCR_CONFIG))

    return "\n".join(chunks).strip()


def extract_text_by_format(file_path: Path, fmt: DocumentInputFormat) -> tuple[str, bool]:
    """
    Returns:
        (text, ocr_used)

    Contract alignment:
    - txt/docx => ocr_used=False
    - pdf => OCR fallback only if native extraction is empty
    - images => OCR always used if text extraction is requested
    """
    if fmt == DocumentInputFormat.txt:
        return extract_text_from_txt(file_path), False

    if fmt == DocumentInputFormat.docx:
        return extract_text_from_docx(file_path), False

    if fmt == DocumentInputFormat.pdf:
        text = extract_text_from_pdf_text(file_path)
        if text:
            return text, False
        return extract_text_from_pdf_ocr(file_path), True

    if fmt in (DocumentInputFormat.jpg, DocumentInputFormat.jpeg, DocumentInputFormat.png):
        return extract_text_from_image(file_path), True

    raise ValueError(f"Unsupported file format: {fmt.value}")


# ----------------------------
# Word-count helpers
# ----------------------------

def count_words(text: str) -> int:
    return len(text.split())


def enforce_word_limit(word_count: int) -> None:
    if word_count < 1:
        raise ValueError("Document contains no words.")
    if word_count > MAX_WORD_COUNT:
        raise ValueError(f"Document exceeds maximum allowed word count of {MAX_WORD_COUNT}.")


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
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("Inline text cannot be empty.")

    if input_format != DocumentInputFormat.txt:
        raise ValueError("Inline text payload must use input_format='txt'.")

    word_count = count_words(normalized)
    enforce_word_limit(word_count)

    metadata = DocumentMetadata(
        input_format=DocumentInputFormat.txt,
        file_size_mb=0.0,
        extracted_word_count=word_count,
        ocr_used=False,
    )
    return DocumentPayload(text=normalized, metadata=metadata)


def build_conversion_document_payload(file_path: str) -> DocumentPayload:
    """
    Build a DocumentPayload for the convert action.

    Contract alignment:
    - convert accepts pdf/docx/jpg/jpeg/png
    - text is optional for convert
    - extracted_word_count is optional for convert
    - no language detection is attached here
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("File not found.")

    fmt = detect_format(path)
    if fmt not in CONVERSION_INPUT_FORMATS:
        raise ValueError("convert only supports: pdf, docx, jpg, jpeg, png.")

    metadata = DocumentMetadata(
        input_format=fmt,
        file_size_mb=get_file_size_mb(path),
        extracted_word_count=None,
        ocr_used=False,  # conversion path does not need OCR/text extraction
    )
    return DocumentPayload(text=None, metadata=metadata)


def build_ai_document_payload(file_path: str) -> DocumentPayload:
    """
    Build a DocumentPayload for AI document actions:
    summarize, grammar_correct, translate, explain, generate_questions, generate_answers

    Contract alignment:
    - only pdf/docx/txt are allowed
    - extracted text is required
    - extracted_word_count is required
    - detected_language is omitted; analyzer/server handles detection
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("File not found.")

    fmt = detect_format(path)
    if fmt not in AI_DOC_INPUT_FORMATS:
        raise ValueError("AI document actions only support input formats: pdf, docx, txt.")

    file_size_mb = get_file_size_mb(path)
    text, ocr_used = extract_text_by_format(path, fmt)

    normalized = text.strip()
    if not normalized:
        raise ValueError("Document text could not be extracted.")

    word_count = count_words(normalized)
    enforce_word_limit(word_count)

    metadata = DocumentMetadata(
        input_format=fmt,
        file_size_mb=file_size_mb,
        extracted_word_count=word_count,
        ocr_used=ocr_used,
    )
    return DocumentPayload(text=normalized, metadata=metadata)


def build_document_payload_for_action(
    *,
    action: FeatureType,
    file_path: Optional[str] = None,
    inline_text: Optional[str] = None,
) -> DocumentPayload:
    """
    Runtime-aware normalization entrypoint.

    Rules:
    - inline_text path => txt payload for AI document actions only
    - upload path + convert => metadata-only payload
    - upload path + AI doc action => extracted-text payload
    """
    if inline_text is not None:
        if file_path is not None:
            raise ValueError("Provide either file_path or inline_text, not both.")
        if action not in AI_DOC_ACTIONS:
            raise ValueError(f"{action.value} does not support inline text through this extraction path.")
        return build_inline_text_payload(inline_text)

    if file_path is None:
        raise ValueError("Either file_path or inline_text must be provided.")

    if action in CONVERSION_ACTIONS:
        return build_conversion_document_payload(file_path)

    if action in AI_DOC_ACTIONS:
        return build_ai_document_payload(file_path)

    raise ValueError(f"Unsupported document action for extraction: {action.value}")
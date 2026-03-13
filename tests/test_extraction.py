import sys
from pathlib import Path

import fitz
import pytest
from PIL import Image
import src.extraction
from src.extraction import (
    AI_DOC_ACTIONS,
    CONVERSION_ACTIONS,
    build_ai_document_payload,
    build_conversion_document_payload,
    build_document_payload_for_action,
    build_inline_text_payload,
    count_words,
    detect_format,
    enforce_word_limit,
    extract_text_by_format,
    extract_text_from_docx,
    extract_text_from_pdf_ocr,
    extract_text_from_pdf_text,
    extract_text_from_txt,
    get_file_size_mb,
    preprocess_for_ocr,
)


def _write_text_file(path: Path, content: str, encoding: str = "utf-8") -> Path:
    path.write_text(content, encoding=encoding)
    return path


def _create_docx(path: Path, paragraphs: list[str]) -> Path:
    doc = src.extraction.docx.Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(path)
    return path


def _create_pdf_with_text(path: Path, text: str) -> Path:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return path


def _create_blank_pdf(path: Path) -> Path:
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    return path


def _create_image(path: Path, size=(40, 40), color=(255, 255, 255)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# ----------------------------
# File helpers
# ----------------------------

def test_preprocess_for_ocr_returns_pil_image():
    image = Image.new("RGB", (20, 20), color=(255, 255, 255))

    processed = preprocess_for_ocr(image)

    assert isinstance(processed, Image.Image)
    assert processed.size == image.size


def test_get_file_size_mb_returns_positive_rounded_size(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"a" * 2048)

    size_mb = get_file_size_mb(file_path)

    assert size_mb > 0
    assert size_mb == round(size_mb, 4)


def test_get_file_size_mb_raises_for_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    with pytest.raises(ValueError, match=r"File is empty\."):
        get_file_size_mb(file_path)


def test_get_file_size_mb_raises_when_file_exceeds_limit(tmp_path, monkeypatch):
    file_path = tmp_path / "large.txt"
    file_path.write_bytes(b"a" * 2048)
    monkeypatch.setattr(src.extraction, "MAX_FILE_SIZE_MB", 0.0001)

    with pytest.raises(ValueError, match=r"File exceeds maximum allowed size of 0\.0001 MB\."):
        get_file_size_mb(file_path)


def test_detect_format_returns_matching_enum(tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"dummy")

    fmt = detect_format(file_path)

    assert fmt == src.extraction.DocumentInputFormat.pdf


def test_detect_format_raises_for_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.xls"
    file_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="Unsupported file format: xls"):
        detect_format(file_path)


# ----------------------------
# Text extraction
# ----------------------------

def test_extract_text_from_txt_returns_stripped_text(tmp_path):
    file_path = _write_text_file(tmp_path / "sample.txt", "  hello world\n")

    text = extract_text_from_txt(file_path)

    assert text == "hello world"


def test_extract_text_from_txt_raises_for_non_utf8(tmp_path):
    file_path = tmp_path / "bad.txt"
    file_path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(ValueError, match=r"TXT file must be valid UTF-8\."):
        extract_text_from_txt(file_path)


def test_extract_text_from_docx_joins_paragraphs(tmp_path):
    file_path = _create_docx(tmp_path / "sample.docx", ["First paragraph", "Second paragraph"])

    text = extract_text_from_docx(file_path)

    assert text == "First paragraph\nSecond paragraph"


def test_extract_text_from_pdf_text_reads_native_text(tmp_path):
    file_path = _create_pdf_with_text(tmp_path / "sample.pdf", "Hello from PDF")

    text = extract_text_from_pdf_text(file_path)

    assert "Hello from PDF" in text


def test_extract_text_from_pdf_ocr_uses_pixmap_pipeline_and_tesseract(tmp_path, monkeypatch):
    file_path = _create_blank_pdf(tmp_path / "scan.pdf")
    captured = {}

    def fake_preprocess(image):
        captured["preprocess_size"] = image.size
        return image

    def fake_ocr(image, lang, config):
        captured["ocr_size"] = image.size
        captured["lang"] = lang
        captured["config"] = config
        return "OCR TEXT"

    monkeypatch.setattr(src.extraction, "preprocess_for_ocr", fake_preprocess)
    monkeypatch.setattr(src.extraction.pytesseract, "image_to_string", fake_ocr)

    text = extract_text_from_pdf_ocr(file_path, zoom=2.0)

    assert text == "OCR TEXT"
    assert captured["preprocess_size"] == captured["ocr_size"]
    assert captured["lang"] == src.extraction.OCR_LANG
    assert captured["config"] == src.extraction.OCR_CONFIG


def test_extract_text_by_format_for_txt_returns_text_without_ocr(tmp_path):
    file_path = _write_text_file(tmp_path / "sample.txt", "hello world")

    text, ocr_used = extract_text_by_format(file_path, src.extraction.DocumentInputFormat.txt)

    assert text == "hello world"
    assert ocr_used is False


def test_extract_text_by_format_for_docx_returns_text_without_ocr(tmp_path):
    file_path = _create_docx(tmp_path / "sample.docx", ["Alpha", "Beta"])

    text, ocr_used = extract_text_by_format(file_path, src.extraction.DocumentInputFormat.docx)

    assert text == "Alpha\nBeta"
    assert ocr_used is False


def test_extract_text_by_format_for_pdf_prefers_native_text(tmp_path, monkeypatch):
    file_path = _create_pdf_with_text(tmp_path / "sample.pdf", "native text")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OCR fallback should not run when native PDF text exists")

    monkeypatch.setattr(src.extraction, "extract_text_from_pdf_ocr", fail_if_called)

    text, ocr_used = extract_text_by_format(file_path, src.extraction.DocumentInputFormat.pdf)

    assert "native text" in text
    assert ocr_used is False


def test_extract_text_by_format_for_pdf_falls_back_to_ocr_when_native_text_is_empty(tmp_path, monkeypatch):
    file_path = _create_blank_pdf(tmp_path / "scan.pdf")
    monkeypatch.setattr(src.extraction, "extract_text_from_pdf_text", lambda path: "")
    monkeypatch.setattr(src.extraction, "extract_text_from_pdf_ocr", lambda path: "ocr output")

    text, ocr_used = extract_text_by_format(file_path, src.extraction.DocumentInputFormat.pdf)

    assert text == "ocr output"
    assert ocr_used is True


@pytest.mark.parametrize(
    "fmt,filename",
    [
        (src.extraction.DocumentInputFormat.jpg, "sample.jpg"),
        (src.extraction.DocumentInputFormat.jpeg, "sample.jpeg"),
        (src.extraction.DocumentInputFormat.png, "sample.png"),
    ],
)
def test_extract_text_by_format_for_images_uses_ocr(tmp_path, monkeypatch, fmt, filename):
    file_path = _create_image(tmp_path / filename)
    monkeypatch.setattr(src.extraction, "extract_text_from_image", lambda path: "image text")

    text, ocr_used = extract_text_by_format(file_path, fmt)

    assert text == "image text"
    assert ocr_used is True


def test_extract_text_by_format_raises_for_unsupported_enum_value():
    class UnsupportedFormat:
        value = "unsupported"

    with pytest.raises(ValueError, match="Unsupported file format: unsupported"):
        extract_text_by_format(Path("unused.bin"), UnsupportedFormat())


# ----------------------------
# Word-count helpers
# ----------------------------

def test_count_words_splits_on_whitespace():
    assert count_words("one   two\nthree\tfour") == 4


def test_enforce_word_limit_raises_for_zero_words():
    with pytest.raises(ValueError, match=r"Document contains no words\."):
        enforce_word_limit(0)


def test_enforce_word_limit_raises_when_over_limit(monkeypatch):
    monkeypatch.setattr(src.extraction, "MAX_WORD_COUNT", 3)

    with pytest.raises(ValueError, match=r"Document exceeds maximum allowed word count of 3\."):
        enforce_word_limit(4)


# ----------------------------
# Payload builders
# ----------------------------

def test_build_inline_text_payload_builds_txt_payload():
    payload = build_inline_text_payload("  hello world  ")

    assert payload.text == "hello world"
    assert payload.metadata.input_format == src.extraction.DocumentInputFormat.txt
    assert payload.metadata.file_size_mb == 0.0
    assert payload.metadata.extracted_word_count == 2
    assert payload.metadata.ocr_used is False


def test_build_inline_text_payload_rejects_empty_text():
    with pytest.raises(ValueError, match=r"Inline text cannot be empty\."):
        build_inline_text_payload("   ")


def test_build_inline_text_payload_rejects_non_txt_input_format():
    with pytest.raises(ValueError, match=r"Inline text payload must use input_format='txt'\."):
        build_inline_text_payload("hello", input_format=src.extraction.DocumentInputFormat.pdf)


def test_build_conversion_document_payload_builds_metadata_only_payload(tmp_path, monkeypatch):
    file_path = _create_image(tmp_path / "image.png")
    monkeypatch.setattr(src.extraction, "get_file_size_mb", lambda path: 1.2345)

    payload = build_conversion_document_payload(str(file_path))

    assert payload.text is None
    assert payload.metadata.input_format == src.extraction.DocumentInputFormat.png
    assert payload.metadata.file_size_mb == 1.2345
    assert payload.metadata.extracted_word_count is None
    assert payload.metadata.ocr_used is False


def test_build_conversion_document_payload_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match=r"File not found\."):
        build_conversion_document_payload(str(missing))


def test_build_conversion_document_payload_raises_for_ai_only_format(tmp_path):
    file_path = _write_text_file(tmp_path / "sample.txt", "hello")

    with pytest.raises(ValueError, match=r"convert only supports: pdf, docx, jpg, jpeg, png\."):
        build_conversion_document_payload(str(file_path))


def test_build_ai_document_payload_builds_full_payload(tmp_path, monkeypatch):
    file_path = _write_text_file(tmp_path / "sample.txt", "hello world from file")
    monkeypatch.setattr(src.extraction, "get_file_size_mb", lambda path: 0.1)
    monkeypatch.setattr(src.extraction, "extract_text_by_format", lambda path, fmt: ("  hello world from file  ", False))

    payload = build_ai_document_payload(str(file_path))

    assert payload.text == "hello world from file"
    assert payload.metadata.input_format == src.extraction.DocumentInputFormat.txt
    assert payload.metadata.file_size_mb == 0.1
    assert payload.metadata.extracted_word_count == 4
    assert payload.metadata.ocr_used is False


def test_build_ai_document_payload_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError, match=r"File not found\."):
        build_ai_document_payload(str(missing))


@pytest.mark.parametrize("filename", ["image.png", "photo.jpg", "photo.jpeg"])
def test_build_ai_document_payload_rejects_non_ai_formats(tmp_path, filename):
    file_path = _create_image(tmp_path / filename)

    with pytest.raises(ValueError, match=r"AI document actions only support input formats: pdf, docx, txt\."):
        build_ai_document_payload(str(file_path))


def test_build_ai_document_payload_raises_when_extracted_text_is_empty(tmp_path, monkeypatch):
    file_path = _write_text_file(tmp_path / "sample.txt", "hello")
    monkeypatch.setattr(src.extraction, "get_file_size_mb", lambda path: 0.1)
    monkeypatch.setattr(src.extraction, "extract_text_by_format", lambda path, fmt: ("   ", False))

    with pytest.raises(ValueError, match=r"Document text could not be extracted\."):
        build_ai_document_payload(str(file_path))


def test_build_document_payload_for_action_rejects_both_inline_text_and_file_path():
    action = next(iter(AI_DOC_ACTIONS))

    with pytest.raises(ValueError, match=r"Provide either file_path or inline_text, not both\."):
        build_document_payload_for_action(action=action, file_path="a.txt", inline_text="hello")


def test_build_document_payload_for_action_rejects_inline_text_for_non_ai_action():
    action = next(iter(CONVERSION_ACTIONS))

    with pytest.raises(ValueError, match=rf"{action.value} does not support inline text through this extraction path\."):
        build_document_payload_for_action(action=action, inline_text="hello")


def test_build_document_payload_for_action_requires_one_input():
    action = next(iter(AI_DOC_ACTIONS))

    with pytest.raises(ValueError, match=r"Either file_path or inline_text must be provided\."):
        build_document_payload_for_action(action=action)


def test_build_document_payload_for_action_routes_inline_text_to_inline_builder(monkeypatch):
    action = next(iter(AI_DOC_ACTIONS))
    sentinel = object()

    def fake_builder(text):
        assert text == "hello"
        return sentinel

    monkeypatch.setattr(src.extraction, "build_inline_text_payload", fake_builder)

    payload = build_document_payload_for_action(action=action, inline_text="hello")

    assert payload is sentinel


def test_build_document_payload_for_action_routes_convert_action_to_conversion_builder(monkeypatch):
    action = next(iter(CONVERSION_ACTIONS))
    sentinel = object()

    def fake_builder(file_path):
        assert file_path == "sample.pdf"
        return sentinel

    monkeypatch.setattr(src.extraction, "build_conversion_document_payload", fake_builder)

    payload = build_document_payload_for_action(action=action, file_path="sample.pdf")

    assert payload is sentinel


def test_build_document_payload_for_action_routes_ai_action_to_ai_builder(monkeypatch):
    action = next(iter(AI_DOC_ACTIONS))
    sentinel = object()

    def fake_builder(file_path):
        assert file_path == "sample.txt"
        return sentinel

    monkeypatch.setattr(src.extraction, "build_ai_document_payload", fake_builder)

    payload = build_document_payload_for_action(action=action, file_path="sample.txt")

    assert payload is sentinel


def test_build_document_payload_for_action_rejects_unsupported_action():
    class UnsupportedAction:
        value = "unsupported_action"

    with pytest.raises(ValueError, match="Unsupported document action for extraction: unsupported_action"):
        build_document_payload_for_action(action=UnsupportedAction(), file_path="sample.txt")

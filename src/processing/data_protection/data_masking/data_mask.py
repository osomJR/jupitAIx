from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import docx
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter

from src.extraction import OCR_CONFIG, preprocess_for_ocr, resolve_ocr_lang
from src.processing.data_protection.client import (
    DEFAULT_DLP_LOCATION,
    DEFAULT_MIN_LIKELIHOOD,
    DetectionCandidate,
    GoogleSDPClient,
    TextFinding,
    build_google_sdp_client,
    inspect_sensitive_text,
    preview_candidates_from_text,
)
from src.schema import (
    AnalyzerRequest,
    AnalyzerResponse,
    DataMaskingRequest,
    DocumentInputFormat,
    DocumentPayload,
    FeatureType,
    SensitiveDataType,
)
from src.validation import build_file_result, validate_analyzer_request, validate_analyzer_response

DEFAULT_PDF_RENDER_SCALE = 2.0

_IMAGE_OUTPUT_MAP: dict[DocumentInputFormat, str] = {
    DocumentInputFormat.jpg: "JPEG",
    DocumentInputFormat.jpeg: "JPEG",
    DocumentInputFormat.png: "PNG",
}

_FILE_OUTPUT_MAP = {
    DocumentInputFormat.pdf: "pdf",
    DocumentInputFormat.docx: "docx",
    DocumentInputFormat.jpg: "jpg",
    DocumentInputFormat.jpeg: "jpeg",
    DocumentInputFormat.png: "png",
}

_SUPPORTED_INPUTS = {
    DocumentInputFormat.pdf,
    DocumentInputFormat.docx,
    DocumentInputFormat.jpg,
    DocumentInputFormat.jpeg,
    DocumentInputFormat.png,
}


@dataclass(frozen=True)
class OCRWord:
    text: str
    bbox: tuple[int, int, int, int]
    start: int
    end: int


def _ocr_words_from_image(image: Image.Image, *, ocr_lang: Optional[str] = None) -> tuple[str, list[OCRWord]]:
    processed = preprocess_for_ocr(image.convert("RGB"))
    data = pytesseract.image_to_data(
        processed,
        lang=ocr_lang or resolve_ocr_lang(),
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    words: list[OCRWord] = []
    text_parts: list[str] = []
    cursor = 0
    texts = data.get("text", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    for idx, raw in enumerate(texts):
        token = (raw or "").strip()
        if not token:
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        end = cursor
        bbox = (
            int(lefts[idx]),
            int(tops[idx]),
            int(lefts[idx]) + int(widths[idx]),
            int(tops[idx]) + int(heights[idx]),
        )
        words.append(OCRWord(text=token, bbox=bbox, start=start, end=end))
    return "".join(text_parts), words


def _normalize_text_for_compare(value: str) -> str:
    import re
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _boxes_for_text_spans(findings: Sequence[TextFinding], words: Sequence[OCRWord]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for finding in findings:
        matched = [w for w in words if not (w.end <= finding.start or w.start >= finding.end)]
        if not matched:
            needle = _normalize_text_for_compare(finding.quote)
            for i in range(len(words)):
                buf = []
                for j in range(i, min(len(words), i + 12)):
                    buf.append(words[j])
                    candidate = _normalize_text_for_compare(" ".join(item.text for item in buf))
                    if candidate == needle:
                        matched = buf
                        break
                if matched:
                    break
        if not matched:
            continue
        x0 = min(w.bbox[0] for w in matched)
        y0 = min(w.bbox[1] for w in matched)
        x1 = max(w.bbox[2] for w in matched)
        y1 = max(w.bbox[3] for w in matched)
        boxes.append((x0, y0, x1, y1))
    unique: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box not in seen:
            unique.append(box)
            seen.add(box)
    return unique


def _iter_table_paragraphs(table: Any):
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                yield paragraph
            for nested in cell.tables:
                yield from _iter_table_paragraphs(nested)


def _iter_document_paragraphs(document: docx.Document):
    for paragraph in document.paragraphs:
        yield paragraph
    for table in document.tables:
        yield from _iter_table_paragraphs(table)
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            yield paragraph
        for table in section.header.tables:
            yield from _iter_table_paragraphs(table)
        for paragraph in section.footer.paragraphs:
            yield paragraph
        for table in section.footer.tables:
            yield from _iter_table_paragraphs(table)


def _clear_paragraph_runs(paragraph: Any) -> None:
    for run in list(paragraph.runs):
        paragraph._element.remove(run._element)


def _set_paragraph_text_preserving_block(paragraph: Any, text: str) -> None:
    _clear_paragraph_runs(paragraph)
    paragraph.add_run(text)


def _output_path(source_path: Path, output_dir: Path, suffix: str = "_masked") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source_path.stem}{suffix}{source_path.suffix.lower()}"


def _ensure_mask_request(request: AnalyzerRequest) -> tuple[DocumentPayload, DataMaskingRequest]:
    if request.action != FeatureType.data_mask:
        raise ValueError("data_mask.py only handles action='data_mask'.")
    if not isinstance(request.input, DocumentPayload):
        raise ValueError("data_mask requires DocumentPayload input.")
    if not isinstance(request.payload, DataMaskingRequest):
        raise ValueError("data_mask requires DataMaskingRequest payload.")
    if request.input.metadata.input_format not in _SUPPORTED_INPUTS:
        raise ValueError("data_mask only supports pdf, docx, jpg, jpeg, png.")
    if request.policy.structure_preservation is not True:
        raise ValueError("structure_preservation must be True for data_mask.")
    return request.input, request.payload


def _build_masking_response(*, request: AnalyzerRequest, output_path: Path) -> AnalyzerResponse:
    input_payload, _ = _ensure_mask_request(request)
    result = build_file_result(
        filename=output_path.name,
        output_format=_FILE_OUTPUT_MAP[input_payload.metadata.input_format],
        file_size_mb=round(output_path.stat().st_size / (1024 * 1024), 4),
        algorithm_version="google-sdp-data-mask-v1",
    )
    response = AnalyzerResponse(
        action=FeatureType.data_mask,
        input_format=input_payload.metadata.input_format,
        policy=request.policy,
        system_language=request.system_language,
        result=result,
    )
    return validate_analyzer_response(response, request=request)


def _mask_digits(value: str, *, keep_last: int = 4, mask_char: str = "X") -> str:
    digits = [i for i, char in enumerate(value) if char.isdigit()]
    if not digits:
        return value
    to_keep = set(digits[-keep_last:]) if keep_last > 0 else set()
    chars = list(value)
    for idx in digits:
        if idx not in to_keep:
            chars[idx] = mask_char
    return "".join(chars)


def _mask_alpha(value: str, *, keep_first: int = 1, mask_char: str = "X") -> str:
    chars = list(value)
    alpha_indices = [i for i, char in enumerate(chars) if char.isalpha()]
    to_keep = set(alpha_indices[:keep_first]) if keep_first > 0 else set()
    for idx in alpha_indices:
        if idx not in to_keep:
            chars[idx] = mask_char
    return "".join(chars)


def _mask_name(value: str) -> str:
    parts = value.split()
    masked = []
    for part in parts:
        masked.append(part[0] + ("X" * (len(part) - 1)) if len(part) > 1 else "X")
    return " ".join(masked) if masked else "X"


def _mask_email(value: str) -> str:
    if "@" not in value:
        return _mask_alpha(value, keep_first=1)
    local, domain = value.split("@", 1)
    if "." in domain:
        head, _, tld = domain.rpartition(".")
        masked_domain = (head[:1] + "X" * max(1, len(head) - 1)) if head else "X"
        return f"{local[:1]}{'X' * max(1, len(local) - 1)}@{masked_domain}.{tld}"
    return f"{local[:1]}{'X' * max(1, len(local) - 1)}@{'X' * max(1, len(domain))}"


def mask_value_by_target(label: str, quote: str) -> str:
    label = (label or "").lower()
    if label in {SensitiveDataType.name.value, "person_name"}:
        return _mask_name(quote)
    if label in {SensitiveDataType.email_address.value, "email_address"}:
        return _mask_email(quote)
    if label in {SensitiveDataType.phone_number.value, "phone_number"}:
        return _mask_digits(quote, keep_last=2)
    if label in {SensitiveDataType.card_number.value, "credit_card_number", SensitiveDataType.account_number.value}:
        return _mask_digits(quote, keep_last=4)
    if label in {
        SensitiveDataType.national_id.value,
        SensitiveDataType.tax_id.value,
        SensitiveDataType.passport_number.value,
        "passport",
    }:
        return _mask_digits(_mask_alpha(quote, keep_first=1), keep_last=2)
    if label in {SensitiveDataType.contact_address.value, "street_address"}:
        stripped = quote.strip()
        return stripped[:6] + ("X" * max(0, len(stripped) - 6)) if len(stripped) > 6 else "X" * len(stripped)
    if label == SensitiveDataType.date_of_birth.value:
        chars = list(quote)
        digits = [i for i, char in enumerate(chars) if char.isdigit()]
        keep = set(digits[-2:])
        for idx in digits:
            if idx not in keep:
                chars[idx] = "X"
        return "".join(chars)
    if label == SensitiveDataType.age.value:
        return _mask_digits(quote, keep_last=0)
    if label == SensitiveDataType.signature.value:
        return "[SIGNATURE_MASKED]"
    return _mask_alpha(_mask_digits(quote, keep_last=2), keep_first=1)


def mask_text_with_findings(text: str, findings: Sequence[TextFinding]) -> str:
    result = text
    for finding in sorted(findings, key=lambda item: item.start, reverse=True):
        replacement = mask_value_by_target(finding.label, finding.quote)
        result = result[:finding.start] + replacement + result[finding.end:]
    return result


def pixelate_boxes(
    image: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    blur_radius: int = 3,
) -> Image.Image:
    out = image.convert("RGB").copy()
    for x0, y0, x1, y1 in boxes:
        region = out.crop((x0, y0, x1, y1))
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        reduced = region.resize((max(1, width // 10), max(1, height // 10)))
        expanded = reduced.resize((width, height))
        expanded = expanded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        out.paste(expanded, (x0, y0, x1, y1))
    return out


def _resolve_sdp(
    *,
    sdp: GoogleSDPClient | None = None,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> GoogleSDPClient:
    if sdp is not None:
        return sdp
    if not project_id:
        raise ValueError("Provide either sdp or project_id.")
    return build_google_sdp_client(
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
    )


def preview_data_mask_candidates(
    request: AnalyzerRequest | Mapping[str, Any],
    *,
    sdp: GoogleSDPClient | None = None,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> list[DetectionCandidate]:
    req = validate_analyzer_request(request)
    input_payload, payload = _ensure_mask_request(req)
    if not input_payload.text:
        raise ValueError(
            "preview_data_mask_candidates requires extracted document text. "
            "Build the request with extraction.py before previewing."
        )
    resolved = _resolve_sdp(
        sdp=sdp,
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
    )
    return preview_candidates_from_text(
        sdp=resolved,
        text=input_payload.text,
        targets=payload.target_data,
        review_exclusions=payload.review_exclusions,
        min_likelihood=min_likelihood,
    )


def _mask_docx(
    *,
    source_path: Path,
    output_path: Path,
    sdp: GoogleSDPClient,
    payload: DataMaskingRequest,
) -> None:
    document = docx.Document(source_path)
    for paragraph in _iter_document_paragraphs(document):
        original = paragraph.text or ""
        if not original.strip():
            continue
        findings = inspect_sensitive_text(
            sdp=sdp,
            text=original,
            targets=payload.target_data,
            review_exclusions=payload.review_exclusions,
        )
        if findings:
            _set_paragraph_text_preserving_block(paragraph, mask_text_with_findings(original, findings))
    document.save(output_path)


def _mask_image_file(
    *,
    source_path: Path,
    output_path: Path,
    sdp: GoogleSDPClient,
    payload: DataMaskingRequest,
    ocr_languages: Optional[Sequence[str]] = None,
) -> None:
    image = Image.open(source_path).convert("RGB")
    page_text, words = _ocr_words_from_image(image, ocr_lang=resolve_ocr_lang(ocr_languages))
    findings = inspect_sensitive_text(
        sdp=sdp,
        text=page_text,
        targets=payload.target_data,
        review_exclusions=payload.review_exclusions,
    )
    boxes = _boxes_for_text_spans(findings, words)
    result = pixelate_boxes(image, boxes)
    fmt = DocumentInputFormat(source_path.suffix.lower().lstrip("."))
    result.save(output_path, format=_IMAGE_OUTPUT_MAP[fmt])


def _mask_pdf(
    *,
    source_path: Path,
    output_path: Path,
    sdp: GoogleSDPClient,
    payload: DataMaskingRequest,
    ocr_languages: Optional[Sequence[str]] = None,
    render_scale: float = DEFAULT_PDF_RENDER_SCALE,
) -> None:
    src = fitz.open(source_path)
    rendered_pages = []
    try:
        for page in src:
            pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale))
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_text, words = _ocr_words_from_image(image, ocr_lang=resolve_ocr_lang(ocr_languages))
            findings = inspect_sensitive_text(
                sdp=sdp,
                text=page_text,
                targets=payload.target_data,
                review_exclusions=payload.review_exclusions,
            )
            boxes = _boxes_for_text_spans(findings, words)
            rendered_pages.append(pixelate_boxes(image, boxes))
        if not rendered_pages:
            raise ValueError("PDF contains no pages.")
        first, *rest = rendered_pages
        first.save(output_path, save_all=True, append_images=rest, format="PDF", resolution=144.0)
    finally:
        src.close()


def apply_data_mask(
    request: AnalyzerRequest | Mapping[str, Any],
    *,
    source_path: str | Path,
    output_dir: str | Path,
    sdp: GoogleSDPClient | None = None,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
    ocr_languages: Optional[Sequence[str]] = None,
) -> AnalyzerResponse:
    req = validate_analyzer_request(request)
    input_payload, payload = _ensure_mask_request(req)
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    output_path = _output_path(source, Path(output_dir), suffix="_masked")
    resolved = _resolve_sdp(
        sdp=sdp,
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
    )

    fmt = input_payload.metadata.input_format
    if fmt == DocumentInputFormat.docx:
        _mask_docx(source_path=source, output_path=output_path, sdp=resolved, payload=payload)
    elif fmt == DocumentInputFormat.pdf:
        _mask_pdf(
            source_path=source,
            output_path=output_path,
            sdp=resolved,
            payload=payload,
            ocr_languages=ocr_languages,
        )
    elif fmt in {DocumentInputFormat.jpg, DocumentInputFormat.jpeg, DocumentInputFormat.png}:
        _mask_image_file(
            source_path=source,
            output_path=output_path,
            sdp=resolved,
            payload=payload,
            ocr_languages=ocr_languages,
        )
    else:
        raise ValueError("Unsupported data_mask input format.")
    return _build_masking_response(request=req, output_path=output_path)


__all__ = [
    "preview_data_mask_candidates",
    "mask_value_by_target",
    "mask_text_with_findings",
    "pixelate_boxes",
    "apply_data_mask",
]

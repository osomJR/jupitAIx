from __future__ import annotations

"""
V1 upload / ingestion layer.

Purpose:
- receive uploaded files from the API layer
- persist them into a stable uploads directory
- build schema-aligned input payloads for analyzer.py
- keep analyzer.py free of request-ingestion and file-saving concerns

Design notes:
- this module does NOT call analyzer.py directly
- this module does NOT perform LLM / ASR / conversion work
- this module is responsible for:
    1) saving uploaded files
    2) building DocumentPayload / MediaPayload from saved files
- convert/transcribe depend on real file paths, so saved-path wiring happens here
- uploaded filenames are treated as untrusted input
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import shutil
import uuid

from fastapi import UploadFile

from src.extraction import (
    build_ai_document_payload,
    build_conversion_document_payload,
    get_file_size_mb,
)
from src.schema import (
    AudioFormat,
    DocumentPayload,
    FeatureType,
    MediaPayload,
    MediaType,
    VideoFormat,
)


UPLOAD_BASE_DIR = Path("uploads")
DOCUMENT_UPLOAD_DIR = UPLOAD_BASE_DIR / "documents"
MEDIA_UPLOAD_DIR = UPLOAD_BASE_DIR / "media"

ALLOWED_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"}
ALLOWED_MEDIA_SUFFIXES = {".mp3", ".mp4", ".mkv", ".mov"}


@dataclass(frozen=True)
class SavedUpload:
    """
    Result of persisting an uploaded file to local storage.
    """

    original_filename: str
    safe_original_filename: str
    stored_filename: str
    stored_path: str
    suffix: str
    mime_type: Optional[str]


class UploadError(ValueError):
    """Raised when uploaded file ingestion fails."""


def ensure_upload_directories() -> None:
    DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(
    upload: UploadFile,
    *,
    category: str,
) -> SavedUpload:
    """
    Persist an uploaded file to disk and return its stored metadata.

    Security rules:
    - never trust the client filename for persistence
    - whitelist extensions by category
    - always save into controlled directories
    - generate the stored filename on the server
    """
    ensure_upload_directories()

    if upload is None:
        raise UploadError("No upload file was provided.")

    original_filename = (upload.filename or "").strip()
    if not original_filename:
        raise UploadError("Uploaded file must have a filename.")

    safe_original_filename = _safe_original_filename(original_filename)

    raw_suffix = Path(original_filename).suffix
    suffix = _validate_upload_suffix(suffix=raw_suffix, category=category)

    stored_filename = f"{uuid.uuid4().hex}{suffix}"

    if category == "documents":
        destination_dir = DOCUMENT_UPLOAD_DIR
    elif category == "media":
        destination_dir = MEDIA_UPLOAD_DIR
    else:
        raise UploadError("category must be either 'documents' or 'media'.")

    destination_path = destination_dir / stored_filename

    try:
        with destination_path.open("wb") as out_file:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, out_file)
    except Exception as exc:
        raise UploadError(f"Failed to persist uploaded file: {exc}") from exc
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    return SavedUpload(
        original_filename=original_filename,
        safe_original_filename=safe_original_filename,
        stored_filename=stored_filename,
        stored_path=str(destination_path),
        suffix=suffix.lstrip("."),
        mime_type=upload.content_type,
    )


def build_uploaded_document_payload(
    *,
    action: FeatureType,
    upload: UploadFile,
) -> DocumentPayload:
    """
    Save an uploaded document and convert it into the correct DocumentPayload.

    Used for:
    - convert
    - summarize
    - grammar_correct
    - translate
    - explain
    - generate_questions
    - generate_answers
    """
    saved = save_uploaded_file(upload, category="documents")

    if action == FeatureType.convert:
        payload = build_conversion_document_payload(saved.stored_path)
    else:
        payload = build_ai_document_payload(saved.stored_path)

    payload.filename = saved.stored_path
    return payload


def build_uploaded_media_payload(
    *,
    upload: UploadFile,
    media_type: MediaType,
    duration_seconds: int,
) -> MediaPayload:
    """
    Save an uploaded media file and build a schema-aligned MediaPayload.

    Used for:
    - transcribe
    """
    saved = save_uploaded_file(upload, category="media")

    file_size_mb = get_file_size_mb(Path(saved.stored_path))
    media_format = _detect_media_format(saved.suffix, media_type)

    return MediaPayload(
        media_type=media_type,
        media_format=media_format,
        file_size_mb=file_size_mb,
        duration_seconds=duration_seconds,
        filename=saved.stored_path,
        mime_type=saved.mime_type,
    )


def _validate_upload_suffix(*, suffix: str, category: str) -> str:
    normalized = suffix.strip().lower()
    if not normalized.startswith("."):
        raise UploadError("Uploaded file must include a valid extension.")

    if category == "documents":
        allowed = ALLOWED_DOCUMENT_SUFFIXES
    elif category == "media":
        allowed = ALLOWED_MEDIA_SUFFIXES
    else:
        raise UploadError("category must be either 'documents' or 'media'.")

    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise UploadError(
            f"Unsupported file extension for {category}: {normalized}. Allowed: {allowed_text}"
        )

    return normalized


def _safe_original_filename(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise UploadError("Uploaded file must have a filename.")

    basename = Path(raw).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip("-._")
    return safe or "upload"


def _detect_media_format(suffix: str, media_type: MediaType):
    normalized = suffix.strip().lower()

    if media_type == MediaType.audio:
        if normalized != AudioFormat.mp3.value:
            raise UploadError("Audio uploads must be mp3.")
        return AudioFormat.mp3

    if normalized == VideoFormat.mp4.value:
        return VideoFormat.mp4
    if normalized == VideoFormat.mkv.value:
        return VideoFormat.mkv
    if normalized == VideoFormat.mov.value:
        return VideoFormat.mov

    raise UploadError("Video uploads must be one of: mp4, mkv, mov.")


__all__ = [
    "UPLOAD_BASE_DIR",
    "DOCUMENT_UPLOAD_DIR",
    "MEDIA_UPLOAD_DIR",
    "ALLOWED_DOCUMENT_SUFFIXES",
    "ALLOWED_MEDIA_SUFFIXES",
    "SavedUpload",
    "UploadError",
    "ensure_upload_directories",
    "save_uploaded_file",
    "build_uploaded_document_payload",
    "build_uploaded_media_payload",
]

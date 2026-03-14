from __future__ import annotations

"""
V1 document writer layer for AI document actions.

Purpose:
- convert processed AI-action text into real downloadable document artifacts
- preserve paragraph order and content structure reasonably
- keep analyzer.py responsible for orchestration, routing, and response building
- keep processing modules responsible for text transformation only

Design notes:
- schema-agnostic: this module creates document artifacts only
- does not call any LLM / ASR / conversion provider
- focuses on valid .pdf and .docx output generation
- .txt inline output is intentionally out of scope for this layer
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Protocol

from docx import Document
from fpdf import FPDF

from src.storage.artifacts import (
    LocalArtifactStorage,
    StorageBackend,
    guess_content_type
)


SUPPORTED_OUTPUT_FORMATS = {"pdf", "docx"}


@dataclass(frozen=True)
class WrittenArtifact:
    """
    Schema-agnostic written artifact descriptor.

    file_path is the persisted artifact location, not the temporary writer path.
    storage_key and download_url are exposed for future orchestration layers.
    """

    file_name: str
    file_extension: str
    file_size_mb: float
    file_path: str
    storage_key: Optional[str] = None
    download_url: Optional[str] = None


class DocumentWriterBackend(Protocol):
    """Provider interface for writing processed text into document artifacts."""

    def write(
        self,
        *,
        content: str,
        output_format: str,
        planned_output_name: str,
    ) -> WrittenArtifact:
        ...


class RealDocumentWriterBackend:
    """
    Real document writer backend.

    Supported outputs:
    - pdf
    - docx

    Goals:
    - preserve paragraph order
    - preserve empty-line paragraph separation reasonably
    - produce valid downloadable document files
    """

    def __init__(self, storage_backend: Optional[StorageBackend] = None) -> None:
        self.storage_backend = storage_backend or LocalArtifactStorage(
            base_dir="artifacts/ai_documents"
        )

    def write(
        self,
        *,
        content: str,
        output_format: str,
        planned_output_name: str,
    ) -> WrittenArtifact:
        normalized_content = _normalize_content(content)
        normalized_output_format = _normalize_output_format(output_format)
        normalized_name = _normalize_file_name(planned_output_name)

        with TemporaryDirectory(prefix="writer-work-") as workdir:
            output_path = Path(workdir) / normalized_name

            if normalized_output_format == "docx":
                self._write_docx(normalized_content, output_path)
            elif normalized_output_format == "pdf":
                self._write_pdf(normalized_content, output_path)
            else:
                raise ValueError(
                    f"Unsupported writer output format: {normalized_output_format}"
                )

            if not output_path.exists():
                raise RuntimeError(
                    "Document writing completed without producing an output file."
                )

            stored = self.storage_backend.persist(
                source_file_path=str(output_path),
                artifact_name=output_path.name,
                content_type=guess_content_type(str(output_path))
            )

            stored_path = Path(stored.stored_path)
            return WrittenArtifact(
                file_name=output_path.name,
                file_extension=normalized_output_format,
                file_size_mb=_get_file_size_mb(stored_path),
                file_path=str(stored_path),
                storage_key=stored.storage_key,
                download_url=stored.download_url,
            )

    def _write_docx(self, content: str, output_path: Path) -> None:
        document = Document()

        for paragraph_text in _split_paragraphs(content):
            document.add_paragraph(paragraph_text)

        document.save(output_path)

    def _write_pdf(self, content: str, output_path: Path) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        paragraphs = _split_paragraphs(content)
        line_height = 6

        for index, paragraph_text in enumerate(paragraphs):
            text = paragraph_text if paragraph_text.strip() else " "
            pdf.multi_cell(0, line_height, text)
            if index < len(paragraphs) - 1:
                pdf.ln(2)

        pdf.output(str(output_path))


class DocumentWriter:
    """
    Stateless writer façade for analyzer integration.

    Responsibilities:
    - validate local writing preconditions
    - delegate actual document creation to a backend
    - return schema-agnostic artifact metadata only
    """

    def __init__(self, backend: Optional[DocumentWriterBackend] = None) -> None:
        self.backend = backend or RealDocumentWriterBackend()

    def write(
        self,
        *,
        content: str,
        output_format: str,
        output_name: str,
    ) -> WrittenArtifact:
        normalized_content = _normalize_content(content)
        normalized_output_format = _normalize_output_format(output_format)
        normalized_name = _normalize_file_name(output_name)

        return self.backend.write(
            content=normalized_content,
            output_format=normalized_output_format,
            planned_output_name=normalized_name,
        )


def write_document(
    *,
    content: str,
    output_format: str,
    output_name: str,
    backend: Optional[DocumentWriterBackend] = None,
) -> WrittenArtifact:
    """Functional convenience wrapper for analyzer integration."""
    writer = DocumentWriter(backend=backend)
    return writer.write(
        content=content,
        output_format=output_format,
        output_name=output_name,
    )


def _split_paragraphs(content: str) -> list[str]:
    """
    Preserve paragraph order reasonably.

    Strategy:
    - normalize line endings
    - split on blank-line boundaries
    """
    normalized = content.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    raw_parts = normalized.split("\\n\\n")

    paragraphs: list[str] = []
    for part in raw_parts:
        cleaned_lines = [line.rstrip() for line in part.split("\\n")]
        paragraph = "\\n".join(cleaned_lines).strip("\\n")
        paragraphs.append(paragraph)

    if not paragraphs:
        return [""]

    return paragraphs


def _normalize_content(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("content must be a string.")
    normalized = content.strip()
    if not normalized:
        raise ValueError("Empty content cannot be written.")
    return normalized


def _normalize_output_format(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_format must be a string.")
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError("output_format must be one of: pdf, docx.")
    return normalized


def _normalize_file_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_name must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("output_name cannot be empty.")
    return normalized


def _get_file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 4)


__all__ = [
    "SUPPORTED_OUTPUT_FORMATS",
    "WrittenArtifact",
    "DocumentWriterBackend",
    "RealDocumentWriterBackend",
    "DocumentWriter",
    "write_document",
]
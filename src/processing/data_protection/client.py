from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from src.schema import SensitiveDataType

DEFAULT_MIN_LIKELIHOOD = "POSSIBLE"
DEFAULT_DLP_LOCATION = "global"


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    quote: str
    occurrences: int
    source: str


@dataclass(frozen=True)
class TextFinding:
    start: int
    end: int
    quote: str
    label: str
    source: str


def _import_dlp_v2():
    try:
        from google.cloud import dlp_v2  # type: ignore
        return dlp_v2
    except Exception as exc:
        raise RuntimeError(
            "Google Sensitive Data Protection client library is required. "
            "Install it with: pip install google-cloud-dlp"
        ) from exc


class GoogleSDPClient:
    """
    Shared Google Sensitive Data Protection wrapper for both redaction and data masking.

    Responsibilities:
    - create/hold the underlying google-cloud-dlp client
    - store project_id and location
    - expose the Google parent resource string
    - expose reusable text inspection helpers shared by redact.py and data_mask.py
    """

    def __init__(
        self,
        project_id: str,
        *,
        location: str = DEFAULT_DLP_LOCATION,
        min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
        client: Any | None = None,
    ) -> None:
        if not project_id or not project_id.strip():
            raise ValueError("project_id is required.")
        if not location or not location.strip():
            raise ValueError("location is required.")

        self.project_id = project_id.strip()
        self.location = location.strip()
        self.min_likelihood = min_likelihood
        self.client = client or _import_dlp_v2().DlpServiceClient()

    @property
    def parent(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}"

    def inspect_text(
        self,
        *,
        text: str,
        targets: Sequence[SensitiveDataType],
        review_exclusions: Sequence[str] = (),
        min_likelihood: Optional[str] = None,
    ) -> list[TextFinding]:
        return inspect_sensitive_text(
            sdp=self,
            text=text,
            targets=targets,
            review_exclusions=review_exclusions,
            min_likelihood=min_likelihood or self.min_likelihood,
        )

    def preview_candidates(
        self,
        *,
        text: str,
        targets: Sequence[SensitiveDataType],
        review_exclusions: Sequence[str] = (),
        min_likelihood: Optional[str] = None,
    ) -> list[DetectionCandidate]:
        return preview_candidates_from_text(
            sdp=self,
            text=text,
            targets=targets,
            review_exclusions=review_exclusions,
            min_likelihood=min_likelihood or self.min_likelihood,
        )


def build_google_sdp_client(
    *,
    project_id: str,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> GoogleSDPClient:
    return GoogleSDPClient(
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
    )


def _coerce_sdp(
    sdp: GoogleSDPClient | None = None,
    *,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> GoogleSDPClient:
    if sdp is not None:
        return sdp
    if client is not None and project_id:
        return build_google_sdp_client(
            project_id=project_id,
            location=location,
            min_likelihood=min_likelihood,
            client=client,
        )
    if project_id:
        return build_google_sdp_client(
            project_id=project_id,
            location=location,
            min_likelihood=min_likelihood,
        )
    raise ValueError("Provide either sdp or project_id.")


def _normalize_text_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _normalized_exclusions(values: Sequence[str]) -> set[str]:
    return {_normalize_text_for_compare(v) for v in values if v and v.strip()}


def _dedupe_findings(findings: Iterable[TextFinding]) -> list[TextFinding]:
    ordered = sorted(findings, key=lambda item: (item.start, -(item.end - item.start), item.label, item.source))
    result: list[TextFinding] = []
    for finding in ordered:
        if result and finding.start >= result[-1].start and finding.end <= result[-1].end:
            continue
        result.append(finding)
    return result


_GOOGLE_INFOTYPES: dict[SensitiveDataType, list[str]] = {
    SensitiveDataType.name: ["PERSON_NAME"],
    SensitiveDataType.email_address: ["EMAIL_ADDRESS"],
    SensitiveDataType.phone_number: ["PHONE_NUMBER"],
    SensitiveDataType.card_number: ["CREDIT_CARD_NUMBER"],
    SensitiveDataType.contact_address: ["STREET_ADDRESS"],
    SensitiveDataType.passport_number: ["PASSPORT"],
}

_LOCAL_REGEX_RULES: dict[SensitiveDataType, list[re.Pattern[str]]] = {
    SensitiveDataType.account_number: [
        re.compile(r"(?i)\b(?:account|acct|a/c)\s*(?:number|no\.?)?\s*[:#-]?\s*([A-Z0-9\-]{6,34})\b")
    ],
    SensitiveDataType.card_number: [
        re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    ],
    SensitiveDataType.national_id: [
        re.compile(r"(?i)\b(?:national\s+id|id\s+number|nin)\s*[:#-]?\s*([A-Z0-9\-]{5,30})\b")
    ],
    SensitiveDataType.tax_id: [
        re.compile(r"(?i)\b(?:tax\s+id|tin|vat(?:\s+number)?)\s*[:#-]?\s*([A-Z0-9\-]{5,30})\b")
    ],
    SensitiveDataType.date_of_birth: [
        re.compile(
            r"(?i)\b(?:dob|date\s+of\s+birth)\s*[:#-]?\s*("
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
            r")\b"
        )
    ],
    SensitiveDataType.age: [
        re.compile(r"(?i)\bage\s*[:#-]?\s*(\d{1,3})\b"),
        re.compile(r"(?i)\b(\d{1,3})\s+years?\s+old\b"),
    ],
    SensitiveDataType.contact_address: [
        re.compile(r"(?i)\b(?:address|contact\s+address)\s*[:#-]?\s*(.+)")
    ],
    SensitiveDataType.signature: [
        re.compile(r"(?i)\b(?:signature|signed\s+by|signatory)\b[:#-]?\s*([A-Za-z][^\n\r]*)?")
    ],
}

_CAPTURED_GROUP_ONLY = {
    SensitiveDataType.account_number,
    SensitiveDataType.national_id,
    SensitiveDataType.tax_id,
    SensitiveDataType.date_of_birth,
    SensitiveDataType.age,
}


def _google_info_types_for_targets(targets: Sequence[SensitiveDataType]) -> list[dict[str, str]]:
    names: list[str] = []
    for target in targets:
        for name in _GOOGLE_INFOTYPES.get(target, []):
            if name not in names:
                names.append(name)
    return [{"name": name} for name in names]


def _local_regex_findings(
    text: str,
    targets: Sequence[SensitiveDataType],
    *,
    exclusions: set[str],
) -> list[TextFinding]:
    findings: list[TextFinding] = []
    for target in targets:
        for pattern in _LOCAL_REGEX_RULES.get(target, []):
            for match in pattern.finditer(text):
                start, end = match.span()
                quote = match.group(0)
                if target in _CAPTURED_GROUP_ONLY and match.lastindex:
                    start, end = match.span(1)
                    quote = match.group(1)
                quote = quote.strip()
                if not quote:
                    continue
                if _normalize_text_for_compare(quote) in exclusions:
                    continue
                findings.append(
                    TextFinding(start=start, end=end, quote=quote, label=target.value, source="local_rule")
                )
    return findings


def _extract_google_span(finding: Any, original_text: str) -> Optional[tuple[int, int]]:
    location = getattr(finding, "location", None)
    if location is None:
        return None

    codepoint_range = getattr(location, "codepoint_range", None)
    if codepoint_range is not None:
        start = int(getattr(codepoint_range, "start", 0))
        end = int(getattr(codepoint_range, "end", 0))
        if end > start:
            return start, end

    byte_range = getattr(location, "byte_range", None)
    if byte_range is not None:
        start = int(getattr(byte_range, "start", 0))
        end = int(getattr(byte_range, "end", 0))
        if end > start:
            prefix = original_text.encode("utf-8")[:start].decode("utf-8", errors="ignore")
            body = original_text.encode("utf-8")[start:end].decode("utf-8", errors="ignore")
            cp_start = len(prefix)
            cp_end = cp_start + len(body)
            if cp_end > cp_start:
                return cp_start, cp_end
    return None


def _google_text_findings(
    *,
    sdp: GoogleSDPClient,
    text: str,
    targets: Sequence[SensitiveDataType],
    exclusions: set[str],
    min_likelihood: str,
) -> list[TextFinding]:
    info_types = _google_info_types_for_targets(targets)
    if not info_types:
        return []

    response = sdp.client.inspect_content(
        request={
            "parent": sdp.parent,
            "inspect_config": {
                "info_types": info_types,
                "include_quote": True,
                "min_likelihood": min_likelihood,
            },
            "item": {"value": text},
        }
    )

    findings: list[TextFinding] = []
    for finding in getattr(getattr(response, "result", None), "findings", []) or []:
        quote = (getattr(finding, "quote", "") or "").strip()
        if not quote:
            continue
        if _normalize_text_for_compare(quote) in exclusions:
            continue
        span = _extract_google_span(finding, text)
        if span is None:
            idx = text.find(quote)
            if idx < 0:
                idx = text.casefold().find(quote.casefold())
            if idx < 0:
                continue
            span = (idx, idx + len(quote))
        info_type = getattr(getattr(finding, "info_type", None), "name", "sensitive_data")
        findings.append(
            TextFinding(
                start=span[0],
                end=span[1],
                quote=quote,
                label=str(info_type).lower(),
                source="google_sdp",
            )
        )
    return findings


def inspect_sensitive_text(
    *,
    text: str,
    targets: Sequence[SensitiveDataType],
    review_exclusions: Sequence[str],
    sdp: GoogleSDPClient | None = None,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> list[TextFinding]:
    resolved = _coerce_sdp(
        sdp,
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
    )
    exclusions = _normalized_exclusions(review_exclusions)
    google_items = _google_text_findings(
        sdp=resolved,
        text=text,
        targets=targets,
        exclusions=exclusions,
        min_likelihood=min_likelihood or resolved.min_likelihood,
    )
    local_items = _local_regex_findings(text, targets, exclusions=exclusions)
    return _dedupe_findings([*google_items, *local_items])


def preview_candidates_from_text(
    *,
    text: str,
    targets: Sequence[SensitiveDataType],
    review_exclusions: Sequence[str],
    sdp: GoogleSDPClient | None = None,
    project_id: str | None = None,
    location: str = DEFAULT_DLP_LOCATION,
    min_likelihood: str = DEFAULT_MIN_LIKELIHOOD,
    client: Any | None = None,
) -> list[DetectionCandidate]:
    grouped: dict[tuple[str, str, str], int] = {}
    for finding in inspect_sensitive_text(
        sdp=sdp,
        project_id=project_id,
        location=location,
        min_likelihood=min_likelihood,
        client=client,
        text=text,
        targets=targets,
        review_exclusions=review_exclusions,
    ):
        key = (finding.label, finding.quote, finding.source)
        grouped[key] = grouped.get(key, 0) + 1

    return [
        DetectionCandidate(label=label, quote=quote, occurrences=count, source=source)
        for (label, quote, source), count in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


__all__ = [
    "DEFAULT_DLP_LOCATION",
    "DEFAULT_MIN_LIKELIHOOD",
    "DetectionCandidate",
    "TextFinding",
    "GoogleSDPClient",
    "build_google_sdp_client",
    "inspect_sensitive_text",
    "preview_candidates_from_text",
]

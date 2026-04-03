"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/components/language_provider";
import {
  ArrowLeft,
  Upload,
  Sparkles,
  XCircle,
  CheckCircle2,
  ShieldCheck,
  EyeOff,
  Download,
} from "lucide-react";
import { commonTranslations, redactPageTranslations } from "@/lib/translations";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".jpg", ".jpeg", ".png"];
const MAX_FILE_SIZE_MB = 10;

const DOCUMENT_TYPES = [
  { value: "invoice", label: "Invoice" },
  { value: "kyc_document", label: "KYC document" },
  { value: "bank_statement", label: "Bank statement" },
  { value: "contract", label: "Contract" },
  { value: "id_document", label: "ID document" },
  { value: "legal_document", label: "Legal document" },
];

const SENSITIVE_LABELS = {
  name: "Name",
  email_address: "Email address",
  phone_number: "Phone number",
  account_number: "Account number",
  card_number: "Card number",
  national_id: "National ID",
  tax_id: "Tax ID",
  passport_number: "Passport number",
  contact_address: "Contact address",
  date_of_birth: "Date of birth",
  age: "Age",
  signature: "Signature",
};

const SUPPORTED_BY_DOCUMENT_TYPE = {
  invoice: [
    "name",
    "email_address",
    "phone_number",
    "account_number",
    "card_number",
    "tax_id",
    "contact_address",
    "signature",
  ],
  kyc_document: [
    "name",
    "email_address",
    "phone_number",
    "account_number",
    "national_id",
    "tax_id",
    "passport_number",
    "contact_address",
    "date_of_birth",
    "age",
    "signature",
  ],
  bank_statement: [
    "name",
    "email_address",
    "phone_number",
    "account_number",
    "card_number",
    "national_id",
    "tax_id",
    "contact_address",
    "date_of_birth",
    "signature",
  ],
  contract: [
    "name",
    "email_address",
    "phone_number",
    "account_number",
    "tax_id",
    "contact_address",
    "signature",
  ],
  id_document: [
    "name",
    "national_id",
    "passport_number",
    "contact_address",
    "date_of_birth",
    "age",
    "signature",
  ],
  legal_document: [
    "name",
    "email_address",
    "phone_number",
    "account_number",
    "card_number",
    "national_id",
    "tax_id",
    "passport_number",
    "contact_address",
    "date_of_birth",
    "age",
    "signature",
  ],
};

function getFileExtension(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
}

function getFileStem(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return filename || "redacted-file";
  return filename.slice(0, lastDot) || "redacted-file";
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function replaceVars(template, vars = {}) {
  return template.replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? "");
}

function pickFirstString(values = []) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function getInputTypeLabel(ext) {
  if (ext === ".pdf") return "PDF";
  if (ext === ".docx") return "DOCX";
  if (ext === ".jpg") return "JPG";
  if (ext === ".jpeg") return "JPEG";
  if (ext === ".png") return "PNG";
  return "Unknown";
}

function parseReviewExclusions(value = "") {
  return value
    .split(/[\n,]/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

function extractResponseMessage(responseData, fallbackMessage = "") {
  return (
    pickFirstString([
      responseData?.detail?.message,
      responseData?.detail?.error,
      responseData?.detail,
      responseData?.message,
      responseData?.error,
      responseData?.result?.message,
      responseData?.data?.message,
      responseData?.analyzer_response?.result?.message,
    ]) || fallbackMessage
  );
}

function extractDownloadInfo(responseData, backendBase, fallbackFilename = "") {
  const artifact = responseData?.artifact || {};
  const result =
    responseData?.analyzer_response?.result || responseData?.result || {};

  const storageKey = pickFirstString([
    artifact?.storage_key,
    artifact?.storageKey,
    result?.storage_key,
    result?.storageKey,
  ]);

  const downloadUrl =
    pickFirstString([
      artifact?.download_url,
      artifact?.downloadUrl,
      result?.download_url,
      result?.downloadUrl,
    ]) ||
    (storageKey
      ? `${backendBase}/api/v1/analyzer/artifacts/${storageKey}`
      : "");

  const filename = pickFirstString([
    artifact?.original_artifact_name,
    artifact?.artifact_name,
    result?.filename,
    fallbackFilename,
  ]);

  const outputFormat = pickFirstString([
    result?.output_format,
    result?.outputFormat,
  ]);

  return {
    storageKey,
    downloadUrl,
    filename,
    outputFormat,
    fileSizeMb: result?.file_size_mb ?? result?.fileSizeMb ?? null,
    contentType: pickFirstString([
      artifact?.content_type,
      artifact?.contentType,
    ]),
  };
}

function extractPreviewUrl(responseData, backendBase) {
  const preview = responseData?.preview_artifact || {};
  const storageKey = pickFirstString([
    preview?.storage_key,
    preview?.storageKey,
  ]);

  return (
    pickFirstString([preview?.download_url, preview?.downloadUrl]) ||
    (storageKey ? `${backendBase}/api/v1/analyzer/artifacts/${storageKey}` : "")
  );
}

function canInlinePreview(ext) {
  return [".pdf", ".jpg", ".jpeg", ".png"].includes(ext);
}

function candidateId(candidate) {
  return (
    candidate?.id ||
    `${candidate?.label || ""}::${candidate?.quote || ""}::${candidate?.source || ""}`
  );
}

function buildMergedReviewExclusions(
  reviewExclusionsText,
  reviewCandidates,
  approvedCandidateIds,
) {
  const manual = parseReviewExclusions(reviewExclusionsText);
  const deselectedQuotes = reviewCandidates
    .filter((candidate) => !approvedCandidateIds.has(candidateId(candidate)))
    .map((candidate) => (candidate?.quote || "").trim())
    .filter(Boolean);

  return [...new Set([...manual, ...deselectedQuotes])];
}

export default function RedactPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t = redactPageTranslations[language] || redactPageTranslations.en;

  const [selectedFile, setSelectedFile] = useState(null);
  const [documentType, setDocumentType] = useState("legal_document");
  const [targetData, setTargetData] = useState(
    SUPPORTED_BY_DOCUMENT_TYPE.legal_document,
  );
  const [reviewExclusionsText, setReviewExclusionsText] = useState("");
  const [error, setError] = useState("");
  const [isReviewing, setIsReviewing] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [resultSummary, setResultSummary] = useState("");
  const [downloadInfo, setDownloadInfo] = useState(null);
  const [stage, setStage] = useState("edit");
  const [reviewCandidates, setReviewCandidates] = useState([]);
  const [approvedCandidateIds, setApprovedCandidateIds] = useState(new Set());
  const [processedPreviewUrl, setProcessedPreviewUrl] = useState("");

  const inputExtension = useMemo(() => {
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [selectedFile]);

  const outputExtension =
    inputExtension || ".pdf / .docx / .jpg / .jpeg / .png";

  const isValidFile = useMemo(() => {
    if (!selectedFile) return false;
    const ext = getFileExtension(selectedFile.name);
    const isAccepted = ACCEPTED_EXTENSIONS.includes(ext);
    const isWithinLimit = selectedFile.size <= MAX_FILE_SIZE_MB * 1024 * 1024;
    return isAccepted && isWithinLimit;
  }, [selectedFile]);

  const supportedTargets = useMemo(() => {
    return SUPPORTED_BY_DOCUMENT_TYPE[documentType] || [];
  }, [documentType]);

  const approvedCount = reviewCandidates.filter((candidate) =>
    approvedCandidateIds.has(candidateId(candidate)),
  ).length;

  const deselectedCount = reviewCandidates.length - approvedCount;

  useEffect(() => {
    setTargetData(SUPPORTED_BY_DOCUMENT_TYPE[documentType] || []);
  }, [documentType]);

  function resetResultState() {
    setResultSummary("");
    setDownloadInfo(null);
    setProcessedPreviewUrl("");
    setReviewCandidates([]);
    setApprovedCandidateIds(new Set());
    setStage("edit");
  }

  function rejectFile(message) {
    setSelectedFile(null);
    setError(message);
    resetResultState();
  }

  function handlePickedFile(file) {
    if (!file) return;

    const ext = getFileExtension(file.name);

    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      rejectFile(
        replaceVars(t.unsupportedFileType, {
          ext: ext || "unknown",
        }),
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      rejectFile(
        replaceVars(t.fileTooLarge, {
          maxSize: MAX_FILE_SIZE_MB,
        }),
      );
      return;
    }

    setError("");
    setSelectedFile(file);
    resetResultState();
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    handlePickedFile(file);
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
    handlePickedFile(file);
  }

  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  function toggleTarget(value) {
    setTargetData((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value],
    );
    setError("");
    resetResultState();
  }

  function toggleCandidate(candidate) {
    const id = candidateId(candidate);
    setApprovedCandidateIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function handleDownload() {
    if (!downloadInfo?.downloadUrl) return;

    const link = document.createElement("a");
    link.href = downloadInfo.downloadUrl;
    link.download = downloadInfo.filename || "redacted-file";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  async function handleProcessAndReview(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError(t.chooseFileToRedact);
      return;
    }

    if (!targetData.length) {
      setError(t.redactionFailed);
      return;
    }

    setIsReviewing(true);
    setError("");
    setResultSummary("");
    setDownloadInfo(null);
    setProcessedPreviewUrl("");

    try {
      const backendBase = process.env.NEXT_PUBLIC_API_BASE_URL;
      if (!backendBase) {
        throw new Error("NEXT_PUBLIC_API_BASE_URL is not set.");
      }

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("document_type", documentType);
      formData.append(
        "system_language",
        language === "fr" ? "french" : "english",
      );

      for (const item of targetData) {
        formData.append("target_data", item);
      }

      const manualExclusions = parseReviewExclusions(reviewExclusionsText);
      for (const item of manualExclusions) {
        formData.append("review_exclusions", item);
      }

      const response = await fetch(
        `${backendBase}/api/v1/analyzer/redact/review`,
        {
          method: "POST",
          body: formData,
          credentials: "include",
        },
      );

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.redactionFailed),
        );
      }

      const resolvedDownload = extractDownloadInfo(
        responseData,
        backendBase,
        `${getFileStem(selectedFile.name)}_redacted${inputExtension}`,
      );

      const previewUrl = extractPreviewUrl(responseData, backendBase);
      const candidates = Array.isArray(responseData?.candidates)
        ? responseData.candidates
        : [];

      setReviewCandidates(candidates);
      setApprovedCandidateIds(new Set(candidates.map(candidateId)));
      setDownloadInfo(resolvedDownload);
      setProcessedPreviewUrl(
        inputExtension === ".docx"
          ? previewUrl
          : resolvedDownload.downloadUrl && canInlinePreview(inputExtension)
            ? resolvedDownload.downloadUrl
            : "",
      );
      setStage("review");

      const summaryLines = [
        t.provisionalReady,
        "",
        `${t.inputFile}: ${selectedFile.name}`,
        `${t.inputExtension}: ${inputExtension}`,
        `${t.outputExtension}: ${inputExtension}`,
        `${t.documentTypeResult}: ${
          DOCUMENT_TYPES.find((item) => item.value === documentType)?.label ||
          documentType
        }`,
        `${t.reviewItemsLabel}: ${candidates.length}`,
        `${t.selectedTargetsLabel} ${targetData.length}`,
        "",
        t.reviewHint,
      ];

      setResultSummary(summaryLines.join("\n"));
    } catch (submitError) {
      setError(submitError?.message || t.redactionFailed);
    } finally {
      setIsReviewing(false);
    }
  }

  async function handleFinalize() {
    if (!selectedFile) {
      setError(t.chooseFileToRedact);
      return;
    }

    setIsFinalizing(true);
    setError("");

    try {
      const backendBase = process.env.NEXT_PUBLIC_API_BASE_URL;
      if (!backendBase) {
        throw new Error("NEXT_PUBLIC_API_BASE_URL is not set.");
      }

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("document_type", documentType);
      formData.append(
        "system_language",
        language === "fr" ? "french" : "english",
      );

      for (const item of targetData) {
        formData.append("target_data", item);
      }

      const exclusions = buildMergedReviewExclusions(
        reviewExclusionsText,
        reviewCandidates,
        approvedCandidateIds,
      );

      for (const item of exclusions) {
        formData.append("review_exclusions", item);
      }

      const response = await fetch(`${backendBase}/api/v1/analyzer/redact`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.redactionFailed),
        );
      }

      const resolvedDownload = extractDownloadInfo(
        responseData,
        backendBase,
        `${getFileStem(selectedFile.name)}_redacted${inputExtension}`,
      );

      const previewUrl = extractPreviewUrl(responseData, backendBase);

      setDownloadInfo(resolvedDownload);
      setProcessedPreviewUrl(
        inputExtension === ".docx"
          ? previewUrl
          : resolvedDownload.downloadUrl && canInlinePreview(inputExtension)
            ? resolvedDownload.downloadUrl
            : "",
      );
      setStage("done");

      const backendMessage = extractResponseMessage(responseData);
      const summaryLines = [
        t.finalReady,
        "",
        `${t.inputFile}: ${selectedFile.name}`,
        `${t.inputExtension}: ${inputExtension}`,
        `${t.outputExtension}: ${inputExtension}`,
        `${t.documentTypeResult}: ${
          DOCUMENT_TYPES.find((item) => item.value === documentType)?.label ||
          documentType
        }`,
        `${t.reviewItemsLabel}: ${reviewCandidates.length}`,
        `${t.approvedCountLabel}: ${approvedCount}`,
        `${t.deselectedCountLabel}: ${deselectedCount}`,
        `${t.exclusionsCount}: ${
          buildMergedReviewExclusions(
            reviewExclusionsText,
            reviewCandidates,
            approvedCandidateIds,
          ).length
        }`,
        `${t.processedFile}: ${
          resolvedDownload.filename ||
          `${getFileStem(selectedFile.name)}_redacted${inputExtension}`
        }`,
      ];

      if (
        resolvedDownload.fileSizeMb !== null &&
        resolvedDownload.fileSizeMb !== undefined
      ) {
        summaryLines.push(`File size: ${resolvedDownload.fileSizeMb} MB`);
      }

      summaryLines.push("");
      summaryLines.push(
        resolvedDownload.downloadUrl ? t.outputReadyText : t.missingDownloadUrl,
      );
      summaryLines.push("");
      summaryLines.push(backendMessage || t.rulesApplied);

      setResultSummary(summaryLines.join("\n"));
    } catch (submitError) {
      setError(submitError?.message || t.redactionFailed);
    } finally {
      setIsFinalizing(false);
    }
  }

  return (
    <main className="app-shell">
      <div className="relative isolate overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_28%),radial-gradient(circle_at_top_right,rgba(168,85,247,0.18),transparent_30%),linear-gradient(to_bottom,#081120,#0a1426,#07111f)]" />

        <div className="relative mx-auto max-w-6xl px-6 py-12 md:px-8 md:py-16">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="mb-8 inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-2 text-sm text-white/80 backdrop-blur transition hover:bg-white/15 hover:text-white"
          >
            <ArrowLeft className="h-4 w-4" />
            {common.back}
          </button>

          <section className="mb-10">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-200 backdrop-blur">
              <Sparkles className="h-4 w-4" />
              {t.badge}
            </div>

            <div className="mt-6 max-w-3xl">
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                {t.title}
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-white/70 md:text-lg">
                {t.description}
              </p>
            </div>
          </section>

          <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <form
              onSubmit={handleProcessAndReview}
              className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl md:p-8"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_left,rgba(34,211,238,0.14),transparent_25%),radial-gradient(circle_at_right,rgba(168,85,247,0.12),transparent_25%)]" />

              <div className="relative">
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className="rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center transition hover:border-white/25 hover:bg-white/10"
                >
                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <Upload className="h-7 w-7 text-cyan-300" />
                  </div>

                  <h2 className="text-lg font-semibold text-white">
                    {t.uploadTitle}
                  </h2>
                  <p className="mt-2 text-sm leading-6 text-white/65">
                    {t.allowedFileInputs}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/55">
                    {t.outputExtensionWillBe}{" "}
                    <span className="font-medium text-white">
                      {outputExtension}
                    </span>
                  </p>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPTED_EXTENSIONS.join(",")}
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-5 rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
                  >
                    {common.chooseFile}
                  </button>
                </div>

                {selectedFile && isValidFile && (
                  <div className="mt-5 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                    <div className="flex items-start gap-3">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
                      <div>
                        <p className="font-medium text-emerald-100">
                          {t.fileAcceptedLabel}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {selectedFile.name} • {formatBytes(selectedFile.size)}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {t.fileTypeLabel} {getInputTypeLabel(inputExtension)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-6 grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <label className="block text-sm font-medium text-white/85">
                      {t.docTypeLabel}
                    </label>
                    <select
                      value={documentType}
                      onChange={(e) => {
                        setDocumentType(e.target.value);
                        setError("");
                        resetResultState();
                      }}
                      className="mt-3 w-full rounded-2xl border border-white/10 bg-[#081120] px-4 py-3 text-sm text-white outline-none focus:border-cyan-300/40"
                    >
                      {DOCUMENT_TYPES.map((item) => (
                        <option
                          key={item.value}
                          value={item.value}
                          className="bg-[#081120]"
                        >
                          {item.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <label className="block text-sm font-medium text-white/85">
                      {t.exclusionsLabel}
                    </label>
                    <textarea
                      value={reviewExclusionsText}
                      onChange={(e) => {
                        setReviewExclusionsText(e.target.value);
                        setError("");
                        resetResultState();
                      }}
                      placeholder={t.exclusionsPlaceholder}
                      rows={5}
                      className="mt-3 w-full rounded-2xl border border-white/10 bg-[#081120] px-4 py-3 text-sm leading-6 text-white outline-none transition placeholder:text-white/30 focus:border-cyan-300/40"
                    />
                  </div>
                </div>

                <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-sm font-medium text-white/85">
                      {t.sensitiveTargetsLabel}
                    </p>

                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setTargetData(supportedTargets);
                          setError("");
                          resetResultState();
                        }}
                        className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80 transition hover:bg-white/10 hover:text-white"
                      >
                        {t.selectAll}
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          setTargetData([]);
                          setError("");
                          resetResultState();
                        }}
                        className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80 transition hover:bg-white/10 hover:text-white"
                      >
                        {t.clearAll}
                      </button>
                    </div>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {supportedTargets.map((item) => {
                      const checked = targetData.includes(item);
                      return (
                        <label
                          key={item}
                          className={`flex cursor-pointer items-center gap-3 rounded-2xl border px-4 py-3 text-sm transition ${
                            checked
                              ? "border-cyan-300/30 bg-cyan-400/10 text-white"
                              : "border-white/10 bg-white/5 text-white/75 hover:bg-white/10"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleTarget(item)}
                            className="h-4 w-4 rounded border-white/20 bg-transparent"
                          />
                          <span>{SENSITIVE_LABELS[item] || item}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>

                {error && (
                  <div className="mt-5 rounded-2xl border border-red-400/20 bg-red-400/10 p-4">
                    <div className="flex items-start gap-3">
                      <XCircle className="mt-0.5 h-5 w-5 text-red-300" />
                      <p className="text-sm leading-6 text-red-100">{error}</p>
                    </div>
                  </div>
                )}

                <div className="mt-6 flex flex-wrap items-center gap-3">
                  <button
                    type="submit"
                    disabled={
                      isReviewing ||
                      isFinalizing ||
                      !selectedFile ||
                      !isValidFile ||
                      !documentType ||
                      targetData.length === 0
                    }
                    className={`rounded-2xl px-5 py-3 text-sm font-semibold transition ${
                      !isReviewing &&
                      !isFinalizing &&
                      selectedFile &&
                      isValidFile &&
                      documentType &&
                      targetData.length > 0
                        ? "bg-white text-slate-900 hover:scale-[1.02] hover:shadow-xl"
                        : "cursor-not-allowed bg-white/10 text-white/40"
                    }`}
                  >
                    {isReviewing ? t.reviewing : t.processAndReview}
                  </button>

                  {stage === "review" && (
                    <button
                      type="button"
                      onClick={handleFinalize}
                      disabled={isFinalizing}
                      className={`rounded-2xl px-5 py-3 text-sm font-semibold transition ${
                        !isFinalizing
                          ? "bg-cyan-300 text-slate-900 hover:scale-[1.02]"
                          : "cursor-not-allowed bg-cyan-300/30 text-slate-900/50"
                      }`}
                    >
                      {isFinalizing ? t.finalizing : t.finalizeAction}
                    </button>
                  )}
                </div>
              </div>
            </form>

            <div className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <EyeOff className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {t.resultTitle}
                    </h2>
                    <p className="text-sm text-white/55">{t.policySubtitle}</p>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-[#081120] p-4">
                  {downloadInfo?.downloadUrl && (
                    <div className="mb-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                      <div className="flex items-start gap-3">
                        <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
                        <div className="flex-1">
                          <p className="font-medium text-emerald-100">
                            {stage === "done"
                              ? t.finalReady
                              : t.provisionalReady}
                          </p>
                          <p className="mt-1 text-sm text-emerald-100/80">
                            {downloadInfo.filename}
                          </p>
                        </div>

                        <button
                          type="button"
                          onClick={handleDownload}
                          className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:scale-[1.02]"
                        >
                          <Download className="h-4 w-4" />
                          {common.download}
                        </button>
                      </div>
                    </div>
                  )}

                  {processedPreviewUrl &&
                    [".pdf", ".docx"].includes(inputExtension) && (
                      <div className="mb-4">
                        <p className="mb-2 text-sm font-medium text-white/80">
                          {t.processedPreviewTitle}
                        </p>
                        <div className="overflow-hidden rounded-2xl border border-white/10 bg-white">
                          <iframe
                            src={processedPreviewUrl}
                            title="Processed preview"
                            className="h-[520px] w-full"
                          />
                        </div>
                      </div>
                    )}

                  {processedPreviewUrl &&
                    [".jpg", ".jpeg", ".png"].includes(inputExtension) && (
                      <div className="mb-4">
                        <p className="mb-2 text-sm font-medium text-white/80">
                          {t.processedPreviewTitle}
                        </p>
                        <div className="overflow-hidden rounded-2xl border border-white/10 bg-white p-2">
                          <image
                            src={processedPreviewUrl}
                            alt="Processed preview"
                            fill
                            unoptimized
                            className="max-h-[520px] w-full rounded-xl object-contain"
                          />
                        </div>
                      </div>
                    )}

                  {!processedPreviewUrl &&
                    inputExtension === ".docx" &&
                    stage !== "edit" && (
                      <div className="mb-4 rounded-2xl border border-amber-400/20 bg-amber-400/10 p-4 text-sm text-amber-100">
                        {t.docxPreviewNotice}
                      </div>
                    )}

                  {stage === "review" && (
                    <div className="mb-4 space-y-4">
                      <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-4">
                        <p className="font-medium text-cyan-100">
                          {t.reviewTitle}
                        </p>
                        <p className="mt-1 text-sm text-cyan-100/80">
                          {t.reviewHint}
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() =>
                            setApprovedCandidateIds(
                              new Set(reviewCandidates.map(candidateId)),
                            )
                          }
                          className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80 transition hover:bg-white/10 hover:text-white"
                        >
                          {t.approveAll}
                        </button>

                        <button
                          type="button"
                          onClick={() => setApprovedCandidateIds(new Set())}
                          className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80 transition hover:bg-white/10 hover:text-white"
                        >
                          {t.clearApproved}
                        </button>
                      </div>

                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                          <p className="text-sm font-medium text-white/85">
                            {t.approvedCountLabel}
                          </p>
                          <p className="mt-1 text-sm text-white/65">
                            {approvedCount}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                          <p className="text-sm font-medium text-white/85">
                            {t.deselectedCountLabel}
                          </p>
                          <p className="mt-1 text-sm text-white/65">
                            {deselectedCount}
                          </p>
                        </div>
                      </div>

                      <div className="max-h-[420px] space-y-3 overflow-y-auto pr-1">
                        {reviewCandidates.length ? (
                          reviewCandidates.map((candidate) => {
                            const checked = approvedCandidateIds.has(
                              candidateId(candidate),
                            );
                            return (
                              <label
                                key={candidateId(candidate)}
                                className={`block rounded-2xl border p-4 transition ${
                                  checked
                                    ? "border-cyan-300/30 bg-cyan-400/10"
                                    : "border-white/10 bg-white/5"
                                }`}
                              >
                                <div className="flex items-start gap-3">
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleCandidate(candidate)}
                                    className="mt-1 h-4 w-4"
                                  />

                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="rounded-full border border-white/10 bg-white/10 px-2 py-1 text-xs text-white/80">
                                        {SENSITIVE_LABELS[candidate.label] ||
                                          candidate.label}
                                      </span>
                                      <span className="text-xs text-white/50">
                                        {t.occurrencesLabel}:{" "}
                                        {candidate.occurrences ?? 1}
                                      </span>
                                    </div>

                                    <p className="mt-2 break-words text-sm leading-6 text-white/85">
                                      {candidate.quote}
                                    </p>
                                  </div>
                                </div>
                              </label>
                            );
                          })
                        ) : (
                          <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-100">
                            {t.noCandidates}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {resultSummary ? (
                    <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                      {resultSummary}
                    </pre>
                  ) : (
                    <p className="text-sm leading-6 text-white/45">
                      {t.previewEmpty}
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <ShieldCheck className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {common.formatPolicy}
                    </h2>
                    <p className="text-sm text-white/55">{t.policySubtitle}</p>
                  </div>
                </div>

                <div className="space-y-3 text-sm leading-6 text-white/70">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.allowedUploadsLabel}
                    </p>
                    <p className="mt-1 text-white/65">{t.allowedFileInputs}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.outputRuleLabel}
                    </p>
                    <p className="mt-1 text-white/65">{t.outputRuleValue}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.selectedTargetsLabel}
                    </p>
                    <p className="mt-1 text-white/65">{targetData.length}</p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

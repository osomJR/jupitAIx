"use client";

import { useLanguage } from "@/components/language_provider";
import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Upload,
  Sparkles,
  XCircle,
  CheckCircle2,
  ShieldCheck,
  Download,
  Scale,
  FileText,
  FileJson,
  ClipboardCheck,
} from "lucide-react";
import {
  commonTranslations,
  compliancePageTranslations,
} from "@/lib/translations";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".jpg", ".jpeg", ".png"];
const MAX_FILE_SIZE_MB = 10;
const COMPLIANCE_PREVIEW_ENDPOINT = "/api/analyzer/compliance/preview";
const COMPLIANCE_ENDPOINT = "/api/analyzer/compliance";


const REPORT_VARIANTS = [
  "human_readable_report",
  "machine_readable_report",
  "annotated_source_output",
];

const REGULATORY_DOMAINS = [
  "privacy",
  "cybersecurity",
  "aml",
  "consumer_protection",
  "public_sector_access_to_information",
  "licensing",
  "registration",
  "sector_regulator_requirements",
];

const NIGERIA_SECTOR_PACKS = [
  "accounting",
  "agriculture",
  "aviation",
  "banking_and_fintech",
  "energy_and_power",
  "health",
  "insurance",
  "law_and_legal",
  "manufacturing",
  "maritime_and_shipping",
  "media",
  "mining",
  "ngo",
  "oil_and_gas",
  "payment_platforms_and_services",
  "pharmaceuticals",
  "sports",
  "tech",
  "telecom",
];

const EXPANDABLE_SECTOR_PACKS = [
  "accounting",
  "agriculture",
  "aviation",
  "banking_and_fintech",
  "energy_and_power",
  "health",
  "insurance",
  "law_and_legal",
  "manufacturing",
  "maritime_and_shipping",
  "media",
  "mining",
  "ngo",
  "oil_and_gas",
  "payment_platforms_and_services",
  "pharmaceuticals",
  "sports",
  "tech",
  "telecom",
];

const COUNTRY_CONFIG = {
  nigeria: {
    labelKey: "nigeria",
    corePack: "core_control_library",
    sectorPacks: NIGERIA_SECTOR_PACKS,
  },
  us: {
    labelKey: "unitedStates",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  uk: {
    labelKey: "unitedKingdom",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  sa: {
    labelKey: "southAfrica",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  canada: {
    labelKey: "canada",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  france: {
    labelKey: "france",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  togo: {
    labelKey: "togo",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
  ghana: {
    labelKey: "ghana",
    corePack: "core_control_library",
    sectorPacks: EXPANDABLE_SECTOR_PACKS,
  },
};

const DEFAULT_JURISDICTION = "nigeria";

function getFileExtension(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
}

function getFileStem(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return filename || "compliance-report";
  return filename.slice(0, lastDot) || "compliance-report";
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function replaceVars(template = "", vars = {}) {
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

function uniqueStrings(values = []) {
  return [...new Set(values.map((item) => item.trim()).filter(Boolean))];
}

function getFileTypeLabel(ext, t) {
  if (ext === ".pdf") return t.pdfDocument;
  if (ext === ".docx") return t.wordDocument;
  if (ext === ".jpg") return t.jpgImage;
  if (ext === ".jpeg") return t.jpegImage;
  if (ext === ".png") return t.pngImage;
  return t.unknownFile;
}

function getReportOutputExtension(reportVariant) {
  return reportVariant === "machine_readable_report" ? "json" : "pdf";
}

function buildFallbackFilename(
  filename = "",
  reportVariant = "human_readable_report",
) {
  const ext = getReportOutputExtension(reportVariant);
  return `${getFileStem(filename)}.compliance.${ext}`;
}

function extractResponseMessage(responseData, fallbackMessage = "") {
  const detail = responseData?.detail;

  if (typeof detail === "string" && detail.trim()) return detail;
  if (typeof detail?.message === "string" && detail.message.trim()) {
    return detail.message.trim();
  }
  if (typeof detail?.error === "string" && detail.error.trim()) {
    return detail.error.trim();
  }

  return (
    pickFirstString([
      responseData?.message,
      responseData?.error,
      responseData?.result?.message,
      responseData?.data?.message,
      responseData?.analyzer_response?.result?.message,
    ]) || fallbackMessage
  );
}

function extractDownloadInfo(responseData, fallbackFilename = "") {
  const artifact =
    responseData?.artifact || responseData?.output_artifact || {};
  const result =
    responseData?.analyzer_response?.result ||
    responseData?.response?.result ||
    responseData?.result ||
    responseData?.data ||
    {};

  const storageKey = pickFirstString([
    artifact?.storage_key,
    artifact?.storageKey,
    result?.storage_key,
    result?.storageKey,
    responseData?.storage_key,
    responseData?.storageKey,
  ]);

  const downloadUrl = pickFirstString([
    artifact?.download_url,
    artifact?.downloadUrl,
    result?.download_url,
    result?.downloadUrl,
    responseData?.download_url,
    responseData?.downloadUrl,
    responseData?.url,
  ]);

  const filename = pickFirstString([
    artifact?.original_artifact_name,
    artifact?.artifact_name,
    artifact?.artifactName,
    result?.filename,
    result?.name,
    responseData?.filename,
    fallbackFilename,
  ]);

  const outputFormat = pickFirstString([
    result?.output_format,
    result?.outputFormat,
    responseData?.output_format,
    responseData?.outputFormat,
  ]);

  const reportVariant = pickFirstString([
    result?.report_variant,
    result?.reportVariant,
    responseData?.report_variant,
    responseData?.reportVariant,
  ]);

  return {
    storageKey,
    downloadUrl,
    filename,
    outputFormat,
    reportVariant,
    fileSizeMb: result?.file_size_mb ?? result?.fileSizeMb ?? null,
    contentType: pickFirstString([
      artifact?.content_type,
      artifact?.contentType,
      result?.content_type,
      result?.contentType,
    ]),
  };
}

function extractComplianceCounts(responseData) {
  const candidates = [
    responseData?.report?.counts,
    responseData?.preview?.report?.counts,
    responseData?.compliance_report?.counts,
    responseData?.data?.report?.counts,
    responseData?.analyzer_response?.report?.counts,
  ].filter(Boolean);

  const counts = candidates[0];

  if (!counts || typeof counts !== "object") {
    return null;
  }

  return {
    passed: counts.passed ?? 0,
    failed: counts.failed ?? 0,
    warning: counts.warning ?? 0,
    missing: counts.missing ?? 0,
    review_required: counts.review_required ?? 0,
  };
}

function SearchableMultiSelect({
  title,
  helpText,
  emptyText,
  examplesText,
  items,
  selectedValues,
  onToggle,
  getLabel,
  searchPlaceholder,
  clearLabel,
  onClear,
  lockedValues = [],
  lockedLabel = "required",
  disabled = false,
}) {
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim().toLowerCase();
  const filteredItems = items.filter((item) =>
    getLabel(item).toLowerCase().includes(normalizedQuery),
  );

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-white/80">{title}</p>
        {selectedValues.length > lockedValues.length && onClear && (
          <button
            type="button"
            disabled={disabled}
            onClick={disabled ? undefined : onClear}
            className={`text-xs font-medium transition ${
              disabled
                ? "cursor-not-allowed text-white/30"
                : "text-cyan-200 hover:text-cyan-100"
            }`}
          >
            {clearLabel}
          </button>
        )}
      </div>

      {helpText && (
        <p className="text-xs leading-5 text-white/50">{helpText}</p>
      )}
      {emptyText && (
        <p className="mt-1 text-xs leading-5 text-cyan-100/70">{emptyText}</p>
      )}
      {examplesText && (
        <p className="mt-1 text-xs leading-5 text-white/40">{examplesText}</p>
      )}

      <div className="mt-3 flex min-h-10 flex-wrap gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2">
        {selectedValues.map((value) => {
          const isLocked = lockedValues.includes(value);
          return (
            <button
              key={value}
              type="button"
              disabled={disabled || isLocked}
              onClick={() => !disabled && !isLocked && onToggle(value)}
              className={`rounded-full border px-3 py-1 text-xs transition ${
                isLocked
                  ? "cursor-not-allowed border-cyan-300/30 bg-cyan-400/15 text-cyan-100"
                  : disabled
                    ? "cursor-not-allowed border-white/10 bg-white/5 text-white/35"
                    : "border-white/10 bg-white/10 text-white/75 hover:border-cyan-300/30 hover:text-cyan-100"
              }`}
            >
              {getLabel(value)}
              {isLocked ? ` · ${lockedLabel}` : " ×"}
            </button>
          );
        })}

        {selectedValues.length === 0 && (
          <span className="py-1 text-xs text-white/35">{emptyText}</span>
        )}
      </div>

      <input
        type="search"
        value={query}
        disabled={disabled}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={searchPlaceholder}
        className={`mt-3 w-full rounded-2xl border border-white/10 px-4 py-2.5 text-sm text-white outline-none transition placeholder:text-white/35 ${
          disabled
            ? "cursor-not-allowed bg-white/5 text-white/35"
            : "bg-white/5 focus:border-cyan-300/40 focus:bg-white/10"
        }`}
      />

      <div className="mt-3 grid max-h-24 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
        {filteredItems.map((item) => {
          const checked = selectedValues.includes(item);
          const isLocked = lockedValues.includes(item);

          return (
            <button
              key={item}
              type="button"
              disabled={disabled || isLocked}
              onClick={() => !disabled && onToggle(item)}
              className={`flex items-center justify-between gap-2 rounded-xl border px-3 py-2 text-left text-xs transition ${
                checked
                  ? "border-cyan-300/40 bg-cyan-300/15 text-cyan-50"
                  : "border-white/10 bg-white/5 text-white/65 hover:bg-white/10"
              } ${disabled || isLocked ? "cursor-not-allowed opacity-80" : ""}`}
            >
              <span>{getLabel(item)}</span>
              {checked && <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function CompliancePage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t =
    compliancePageTranslations[language] || compliancePageTranslations.en;

  const [jurisdiction, setJurisdiction] = useState(DEFAULT_JURISDICTION);
  const selectedCountryConfig =
    COUNTRY_CONFIG[jurisdiction] || COUNTRY_CONFIG[DEFAULT_JURISDICTION];

  const countryLabels = t.countryLabels || {};
  const selectedCountryLabel =
    countryLabels[selectedCountryConfig.labelKey] || jurisdiction;

  const [selectedFile, setSelectedFile] = useState(null);
  const [sectorPacks, setSectorPacks] = useState([
    COUNTRY_CONFIG[DEFAULT_JURISDICTION].corePack,
  ]);
  const [regulatoryDomains, setRegulatoryDomains] = useState([]);
  const [reportVariant, setReportVariant] = useState("human_readable_report");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewMarkdown, setPreviewMarkdown] = useState("");
  const [previewReport, setPreviewReport] = useState(null);
  const [resultSummary, setResultSummary] = useState("");
  const [downloadInfo, setDownloadInfo] = useState(null);
  const [counts, setCounts] = useState(null);

  const availableSectorPacks = useMemo(() => {
    return [
      selectedCountryConfig.corePack,
      ...selectedCountryConfig.sectorPacks,
    ];
  }, [selectedCountryConfig]);

  const inputExtension = useMemo(() => {
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [selectedFile]);

  const outputExtension = useMemo(
    () => getReportOutputExtension(reportVariant),
    [reportVariant],
  );

  const reportVariantDescription =
    t.reportVariantDescriptions?.[reportVariant] || "";

  const selectedSectorLabels = useMemo(() => {
    return sectorPacks
      .map((pack) => t.sectorPackLabels?.[pack] || pack)
      .join(", ");
  }, [sectorPacks, t]);

  const selectedDomainLabels = useMemo(() => {
    if (!regulatoryDomains.length) return t.allDomains;

    return regulatoryDomains
      .map((domain) => t.regulatoryDomainLabels?.[domain] || domain)
      .join(", ");
  }, [regulatoryDomains, t]);

  const isValidFile = useMemo(() => {
    if (!selectedFile) return false;

    const ext = getFileExtension(selectedFile.name);
    const isAccepted = ACCEPTED_EXTENSIONS.includes(ext);
    const isWithinLimit = selectedFile.size <= MAX_FILE_SIZE_MB * 1024 * 1024;

    return isAccepted && isWithinLimit;
  }, [selectedFile]);

  const canPreview =
    !isPreviewing &&
    !isSubmitting &&
    !!selectedFile &&
    isValidFile &&
    sectorPacks.includes(selectedCountryConfig.corePack) &&
    REPORT_VARIANTS.includes(reportVariant);

  const canGenerate = canPreview && !!previewMarkdown;
  const isProcessing = isPreviewing || isSubmitting;
  
    function resetResultState() {
      setResultSummary("");
      setDownloadInfo(null);
      setCounts(null);
      setPreviewMarkdown("");
      setPreviewReport(null);
    }

  function rejectFile(message) {
    setSelectedFile(null);
    setError(message);
    resetResultState();
  }

  function handlePickedFile(file) {
    if (isProcessing) return;
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
    if (isProcessing) return;

    const file = event.dataTransfer.files?.[0];
    handlePickedFile(file);
  }

  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  function handleJurisdictionChange(nextJurisdiction) {
    if (isProcessing) return;
    const nextConfig =
      COUNTRY_CONFIG[nextJurisdiction] || COUNTRY_CONFIG[DEFAULT_JURISDICTION];

    setJurisdiction(nextJurisdiction);
    setSectorPacks([nextConfig.corePack]);
    setRegulatoryDomains([]);
    setError("");
    resetResultState();
  }

  function toggleSectorPack(value) {
    if (isProcessing) return;
    const corePack = selectedCountryConfig.corePack;

    setSectorPacks((current) => {
      if (value === corePack) {
        return current.includes(corePack) ? current : [corePack, ...current];
      }

      if (current.includes(value)) {
        return current.filter((item) => item !== value);
      }

      return uniqueStrings([corePack, ...current, value]);
    });

    setError("");
    resetResultState();
  }

  function toggleRegulatoryDomain(value) {
    if (isProcessing) return;
    setRegulatoryDomains((current) => {
      if (current.includes(value)) {
        return current.filter((item) => item !== value);
      }

      return [...current, value];
    });

    setError("");
    resetResultState();
  }

  function clearRegulatoryDomains() {
    if (isProcessing) return;
    setRegulatoryDomains([]);
    setError("");
    resetResultState();
  }
  function getArtifactDownloadUrl(info) {
    if (!info) return "";

    if (info.downloadUrl) {
      return info.downloadUrl.replace(
        /^\/api\/v1\/analyzer\/artifacts\//,
        "/api/analyzer/artifacts/",
      );
    }

    if (info.storageKey) {
      return `/api/analyzer/artifacts/${info.storageKey}`;
    }

    return "";
  }
  function handleDownload() {
    const url = getArtifactDownloadUrl(downloadInfo);

    if (!url) {
      setError("Download URL is missing.");
      return;
    }

    const link = document.createElement("a");
    link.href = url;
    link.download = downloadInfo.filename || "compliance-report";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  async function buildComplianceFormData() {
    const formData = new FormData();

    const buffer = await selectedFile.arrayBuffer();

    const fileBlob = new Blob([buffer], {
      type: selectedFile.type || "application/octet-stream",
    });

    formData.append("file", fileBlob, selectedFile.name);
    formData.append("jurisdiction", jurisdiction);
    formData.append("report_variant", reportVariant);
    formData.append("require_human_review", "true");
    formData.append(
      "system_language",
      language === "fr" ? "french" : "english",
    );

    for (const sectorPack of sectorPacks) {
      formData.append("sector_packs", sectorPack);
    }

    for (const regulatoryDomain of regulatoryDomains) {
      formData.append("regulatory_domains", regulatoryDomain);
    }

    return formData;
  }

  async function handlePreview(event) {
    event?.preventDefault();

    if (!selectedFile) {
      setError(t.chooseFileToCheck);
      return;
    }

    if (!sectorPacks.includes(selectedCountryConfig.corePack)) {
      setError(
        replaceVars(t.corePackRequired, {
          country: selectedCountryLabel,
        }),
      );
      return;
    }

    setIsPreviewing(true);
    setError("");
    setResultSummary("");
    setDownloadInfo(null);
    setCounts(null);
    setPreviewMarkdown("");
    setPreviewReport(null);

    try {
      const response = await fetch(COMPLIANCE_PREVIEW_ENDPOINT, {
        method: "POST",
        body: await buildComplianceFormData(),
        credentials: "include",
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.complianceFailed),
        );
      }

      const previewText =
        responseData?.preview_markdown || responseData?.previewMarkdown || "";

      setPreviewMarkdown(previewText);
      setPreviewReport(responseData?.report || null);
      setCounts(extractComplianceCounts(responseData));

      setResultSummary(
        previewText ||
          [
            "Compliance preview completed.",
            "",
            `${t.inputFile}: ${selectedFile.name}`,
            `${t.jurisdictionResult}: ${selectedCountryLabel}`,
            `${t.sectorPacksResult}: ${selectedSectorLabels}`,
            "",
            t.humanReviewRequired,
          ].join("\n"),
      );
    } catch (previewError) {
      setError(previewError?.message || t.complianceFailed);
    } finally {
      setIsPreviewing(false);
    }
  }

  async function handleSubmit(event) {
    event?.preventDefault();

    if (!selectedFile) {
      setError(t.chooseFileToCheck);
      return;
    }

    if (!sectorPacks.includes(selectedCountryConfig.corePack)) {
      setError(
        replaceVars(t.corePackRequired, {
          country: selectedCountryLabel,
        }),
      );
      return;
    }

    setIsSubmitting(true);
    setError("");
    setDownloadInfo(null);

    try {
      const fallbackFilename = buildFallbackFilename(
        selectedFile.name,
        reportVariant,
      );

      const response = await fetch(COMPLIANCE_ENDPOINT, {
        method: "POST",
        body: await buildComplianceFormData(),
        credentials: "include",
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.complianceFailed),
        );
      }

      const resolvedDownload = extractDownloadInfo(
        responseData,
        fallbackFilename,
      );
      const resolvedCounts = extractComplianceCounts(responseData);

      setDownloadInfo(resolvedDownload);
      setCounts(resolvedCounts);

      const reportVariantLabel =
        t.reportVariantLabels?.[reportVariant] || reportVariant;

      const outputFormatLabel =
        t.outputFormatLabels?.[outputExtension] || `.${outputExtension}`;

      const summaryLines = [
        t.complianceCompleted,
        "",
        `${t.inputFile}: ${selectedFile.name}`,
        `${t.inputExtension}: ${inputExtension}`,
        `${t.jurisdictionResult}: ${selectedCountryLabel}`,
        `${t.sectorPacksResult}: ${selectedSectorLabels}`,
        `${t.regulatoryDomainsResult}: ${selectedDomainLabels}`,
        `${t.reportVariantResult}: ${reportVariantLabel}`,
        `${t.outputFormatResult}: ${outputFormatLabel}`,
        `${t.reportFile}: ${resolvedDownload.filename || fallbackFilename}`,
      ];

      if (resolvedCounts) {
        summaryLines.push(
          "",
          t.findingsSummary,
          `${t.passed}: ${resolvedCounts.passed}`,
          `${t.failed}: ${resolvedCounts.failed}`,
          `${t.warning}: ${resolvedCounts.warning}`,
          `${t.missing}: ${resolvedCounts.missing}`,
          `${t.reviewRequiredCount}: ${resolvedCounts.review_required}`,
        );
      }

      summaryLines.push(
        "",
        resolvedDownload.downloadUrl ? t.outputReadyText : t.missingDownloadUrl,
        "",
        t.humanReviewRequired,
      );

      setResultSummary(summaryLines.join("\n"));
    } catch (submitError) {
      setError(submitError?.message || t.complianceFailed);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell min-h-screen overflow-x-hidden">
      <div className="relative isolate min-h-screen overflow-x-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_28%),radial-gradient(circle_at_top_right,rgba(168,85,247,0.18),transparent_30%),linear-gradient(to_bottom,#081120,#0a1426,#07111f)]" />

        <div className="relative mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-4 md:px-5 lg:py-4">
          <header className="mb-3 shrink-0">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <button
                type="button"
                onClick={() => router.push("/")}
                className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-2 text-sm text-white/80 backdrop-blur transition hover:bg-white/15 hover:text-white"
              >
                <ArrowLeft className="h-4 w-4" />
                {common.back}
              </button>

              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-200 backdrop-blur">
                <Sparkles className="h-4 w-4" />
                {t.badge}
              </div>
            </div>

            <div className="mt-3">
              <h1 className="max-w-full text-2xl font-semibold tracking-tight text-white sm:text-3xl lg:whitespace-nowrap lg:text-[2.15rem] lg:leading-tight xl:text-[2.35rem]">
                {t.title}
              </h1>
              <p className="mt-1 max-w-4xl text-sm leading-5 text-white/70 md:text-base">
                {t.description}
              </p>
            </div>
          </header>

          <section className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,0.92fr)_minmax(420px,1.08fr)] lg:items-stretch">
            <form
              onSubmit={handlePreview}
              className="relative min-h-0 overflow-y-auto rounded-3xl border border-white/10 bg-white/8 p-3 backdrop-blur-xl md:p-4 lg:max-h-[calc(100vh-8.5rem)]"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_left,rgba(34,211,238,0.14),transparent_25%),radial-gradient(circle_at_right,rgba(168,85,247,0.12),transparent_25%)]" />

              <div className="relative flex h-full min-h-0 flex-col">
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className="rounded-2xl border border-dashed border-white/15 bg-white/5 p-4 text-center transition hover:border-white/25 hover:bg-white/10 md:p-5"
                >
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <Upload className="h-5 w-5 text-cyan-300" />
                  </div>

                  <h2 className="text-base font-semibold text-white">
                    {t.uploadTitle}
                  </h2>

                  <p className="mt-1 text-xs leading-5 text-white/50">
                    {t.allowedFileInputs}
                  </p>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.jpg,.jpeg,.png"
                    disabled={isProcessing}
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  <button
                    type="button"
                    disabled={isProcessing}
                    onClick={() =>
                      !isProcessing && fileInputRef.current?.click()
                    }
                    className={`mt-3 rounded-2xl px-4 py-2.5 text-sm font-semibold transition ${
                      isProcessing
                        ? "cursor-not-allowed bg-white/10 text-white/35"
                        : "bg-white text-slate-900 hover:scale-[1.02] hover:shadow-xl"
                    }`}
                  >
                    {common.chooseFile}
                  </button>
                </div>

                {selectedFile && isValidFile && (
                  <div className="mt-3 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-3">
                    <div className="flex items-start gap-3">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
                      <div className="min-w-0">
                        <p className="font-medium text-emerald-100">
                          {common.fileAccepted}
                        </p>
                        <p className="mt-1 truncate text-sm text-emerald-100/80">
                          {selectedFile.name} • {formatBytes(selectedFile.size)}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {t.detectedType} {getFileTypeLabel(inputExtension, t)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <label className="block">
                    <span className="mb-2 block text-sm font-medium text-white/80">
                      {t.jurisdictionLabel}
                    </span>
                    <select
                      value={jurisdiction}
                      disabled={isProcessing}
                      onChange={(event) =>
                        handleJurisdictionChange(event.target.value)
                      }
                      className={`w-full rounded-2xl border border-white/10 px-4 py-2.5 text-sm text-white outline-none transition ${
                        isProcessing
                          ? "cursor-not-allowed bg-white/5 text-white/35"
                          : "bg-white/5 focus:border-cyan-300/40 focus:bg-white/10"
                      }`}
                    >
                      {Object.entries(COUNTRY_CONFIG).map(([value, config]) => (
                        <option
                          key={value}
                          value={value}
                          className="bg-slate-900 text-white"
                        >
                          {countryLabels[config.labelKey] || value}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs leading-5 text-white/45">
                      {t.jurisdictionHelp}
                    </p>
                    <p className="mt-1 text-xs leading-5 text-white/35">
                      {t.jurisdictionExamples}
                    </p>
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-sm font-medium text-white/80">
                      {t.reportVariantLabel}
                    </span>
                    <select
                      value={reportVariant}
                      disabled={isProcessing}
                      onChange={(event) => {
                        setReportVariant(event.target.value);
                        setError("");
                        resetResultState();
                      }}
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-white outline-none transition focus:border-cyan-300/40 focus:bg-white/10"
                    >
                      {REPORT_VARIANTS.map((variant) => (
                        <option
                          key={variant}
                          value={variant}
                          className="bg-slate-900 text-white"
                        >
                          {t.reportVariantLabels?.[variant] || variant}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs leading-5 text-white/45">
                      {t.reportVariantHelp}
                    </p>
                    {reportVariantDescription && (
                      <p className="mt-1 rounded-xl border border-cyan-300/20 bg-cyan-400/10 px-3 py-2 text-xs leading-5 text-cyan-100/80">
                        {reportVariantDescription}
                      </p>
                    )}
                    <p className="mt-1 text-xs leading-5 text-white/35">
                      {t.reportVariantExamples}
                    </p>
                  </label>
                </div>

                <div className="mt-3 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-3">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-cyan-300" />
                    <div className="min-w-0 flex-1">
                      <SearchableMultiSelect
                        title={t.sectorPacksLabel}
                        disabled={isProcessing}
                        helpText={replaceVars(t.corePackHelp, {
                          country: selectedCountryLabel,
                        })}
                        emptyText={t.sectorPacksEmptyHelp}
                        examplesText={t.sectorPacksExamples}
                        items={availableSectorPacks}
                        selectedValues={sectorPacks}
                        onToggle={toggleSectorPack}
                        getLabel={(pack) => t.sectorPackLabels?.[pack] || pack}
                        searchPlaceholder={t.searchSectorPacksPlaceholder}
                        clearLabel={t.clearSectorPacks}
                        lockedValues={[selectedCountryConfig.corePack]}
                        lockedLabel={t.requiredLabel}
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-3 rounded-2xl border border-white/10 bg-white/5 p-3">
                  <SearchableMultiSelect
                    title={t.regulatoryDomainsLabel}
                    disabled={isProcessing}
                    helpText={t.regulatoryDomainsHelp}
                    emptyText={t.regulatoryDomainsEmptyHelp}
                    examplesText={t.regulatoryDomainsExamples}
                    items={REGULATORY_DOMAINS}
                    selectedValues={regulatoryDomains}
                    onToggle={toggleRegulatoryDomain}
                    getLabel={(domain) =>
                      t.regulatoryDomainLabels?.[domain] || domain
                    }
                    searchPlaceholder={t.searchRegulatoryDomainsPlaceholder}
                    clearLabel={t.clearDomains}
                    onClear={clearRegulatoryDomains}
                  />
                </div>

                {error && (
                  <div className="mt-3 rounded-2xl border border-red-400/20 bg-red-400/10 p-3">
                    <div className="flex items-start gap-3">
                      <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-300" />
                      <p className="text-sm leading-6 text-red-100">{error}</p>
                    </div>
                  </div>
                )}

                <div className="mt-auto pt-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="submit"
                      disabled={!canPreview}
                      className={`rounded-2xl px-5 py-2.5 text-sm font-semibold transition ${
                        canPreview
                          ? "bg-white text-slate-900 hover:scale-[1.02] hover:shadow-xl"
                          : "cursor-not-allowed bg-white/10 text-white/40"
                      }`}
                    >
                      {isPreviewing ? "Previewing..." : "Preview compliance"}
                    </button>

                    <button
                      type="button"
                      disabled={!canGenerate}
                      onClick={handleSubmit}
                      className={`rounded-2xl px-5 py-2.5 text-sm font-semibold transition ${
                        canGenerate
                          ? "border border-cyan-300/30 bg-cyan-400/10 text-cyan-100 hover:bg-cyan-400/15"
                          : "cursor-not-allowed border border-white/10 bg-white/5 text-white/35"
                      }`}
                    >
                      {isSubmitting ? t.checking : "Generate downloadable file"}
                    </button>

                    {downloadInfo?.downloadUrl && (
                      <button
                        type="button"
                        onClick={handleDownload}
                        className="inline-flex items-center gap-2 rounded-2xl border border-emerald-300/30 bg-emerald-400/10 px-5 py-2.5 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/15"
                      >
                        <Download className="h-4 w-4" />
                        {common.download}
                      </button>
                    )}
                  </div>

                  <div className="mt-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/55">
                    {t.complianceLabel}{" "}
                    <span className="font-medium text-white/85">
                      {selectedCountryLabel}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="min-h-0 lg:h-full">
              <div className="flex min-h-[360px] flex-col rounded-3xl border border-white/10 bg-white/8 p-4 backdrop-blur-xl md:p-5 lg:min-h-[calc(100vh-8.5rem)] lg:max-h-[calc(100vh-8.5rem)]">
                <div className="flex items-center justify-between gap-3">
                  <h2 className="text-lg font-semibold text-white">
                    {t.complianceOutput}
                  </h2>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/55">
                    {t.complianceLabel} {selectedCountryLabel}
                  </span>
                </div>

                {counts && (
                  <div className="mt-3 grid grid-cols-5 gap-2 text-center text-xs">
                    <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-2 text-emerald-100">
                      <p className="font-semibold">{counts.passed}</p>
                      <p className="mt-1 text-[10px] opacity-80">{t.passed}</p>
                    </div>
                    <div className="rounded-2xl border border-red-400/20 bg-red-400/10 p-2 text-red-100">
                      <p className="font-semibold">{counts.failed}</p>
                      <p className="mt-1 text-[10px] opacity-80">{t.failed}</p>
                    </div>
                    <div className="rounded-2xl border border-amber-400/20 bg-amber-400/10 p-2 text-amber-100">
                      <p className="font-semibold">{counts.warning}</p>
                      <p className="mt-1 text-[10px] opacity-80">{t.warning}</p>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-2 text-white/75">
                      <p className="font-semibold">{counts.missing}</p>
                      <p className="mt-1 text-[10px] opacity-80">{t.missing}</p>
                    </div>
                    <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-2 text-cyan-100">
                      <p className="font-semibold">{counts.review_required}</p>
                      <p className="mt-1 text-[10px] opacity-80">
                        {t.reviewRequiredShort}
                      </p>
                    </div>
                  </div>
                )}

                <div className="mt-3 min-h-[320px] flex-1 overflow-y-auto rounded-2xl border border-white/10 bg-[#081120] p-4 lg:max-h-none">
                  {resultSummary ? (
                    <div className="flex h-full min-h-0 flex-col gap-3">
                      <pre className="whitespace-pre-wrap break-words pr-1 text-xs leading-6 text-white/80 md:text-sm">
                        {resultSummary}
                      </pre>

                      {getArtifactDownloadUrl(downloadInfo) && (
                        <div className="shrink-0 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-3">
                          <div className="flex items-start gap-3">
                            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
                            <div className="min-w-0">
                              <p className="font-medium text-emerald-100">
                                {t.downloadReady}
                              </p>
                              <p className="mt-1 truncate text-sm text-emerald-100/80">
                                {downloadInfo.filename}
                              </p>
                              <button
                                type="button"
                                onClick={handleDownload}
                                className="mt-3 inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
                              >
                                <Download className="h-4 w-4" />
                                {common.download}
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex h-full min-h-[180px] items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-4 text-center">
                      <div>
                        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
                          {reportVariant === "machine_readable_report" ? (
                            <FileJson className="h-5 w-5 text-cyan-300" />
                          ) : reportVariant === "annotated_source_output" ? (
                            <ClipboardCheck className="h-5 w-5 text-cyan-300" />
                          ) : (
                            <FileText className="h-5 w-5 text-cyan-300" />
                          )}
                        </div>
                        <p className="max-w-sm text-sm leading-6 text-white/45">
                          {t.previewText}
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="mt-3 grid gap-2 text-xs text-white/45 sm:grid-cols-3">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <p className="font-medium text-white/65">{t.outputTitle}</p>
                    <p className="mt-1">
                      {reportVariant === "machine_readable_report"
                        ? ".json"
                        : ".pdf"}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <p className="font-medium text-white/65">{t.reviewTitle}</p>
                    <p className="mt-1">{t.reviewValue}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <p className="font-medium text-white/65">{t.scopeTitle}</p>
                    <p className="mt-1">{t.scopeValue}</p>
                  </div>
                </div>
              </div>
            </aside>
          </section>
        </div>
      </div>
    </main>
  );
}
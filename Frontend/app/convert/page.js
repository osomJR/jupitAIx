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
  FileType,
  RefreshCw,
  FileText,
  Image as ImageIcon,
  Repeat,
  Download,
} from "lucide-react";
import {
  commonTranslations,
  convertPageTranslations,
} from "@/lib/translations";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".jpg", ".jpeg", ".png"];
const MAX_FILE_SIZE_MB = 10;

function getFileExtension(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
}

function getFileStem(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return filename || "converted-file";
  return filename.slice(0, lastDot) || "converted-file";
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getAllowedOutputExtensions(inputExtension) {
  switch (inputExtension) {
    case ".pdf":
      return [".docx"];
    case ".docx":
      return [".pdf"];
    case ".jpg":
    case ".jpeg":
      return [".pdf", ".docx"];
    case ".png":
      return [".jpg", ".jpeg"];
    default:
      return [];
  }
}

function getInputTypeLabel(ext, t) {
  if (ext === ".pdf") return t.pdfDocument;
  if (ext === ".docx") return t.wordDocument;
  if (ext === ".jpg" || ext === ".jpeg") return t.jpgImage;
  if (ext === ".png") return t.pngImage;
  return t.unknownFile;
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

function buildOutputFilename(filename = "", outputExtension = "") {
  const safeExtension = outputExtension
    ? outputExtension.startsWith(".")
      ? outputExtension
      : `.${outputExtension}`
    : "";

  return `${getFileStem(filename)}${safeExtension}`;
}

function extractArtifactMetadata(responseData) {
  const candidates = [
    responseData?.artifact,
    responseData?.output_artifact,
    responseData?.result?.artifact,
    responseData?.data?.artifact,
    responseData?.result,
    responseData?.data,
    responseData,
  ].filter(Boolean);

  let downloadUrl = "";
  let storageKey = "";
  let artifactName = "";
  let contentType = "";

  for (const candidate of candidates) {
    downloadUrl ||= pickFirstString([
      candidate?.download_url,
      candidate?.downloadUrl,
      candidate?.url,
      candidate?.href,
    ]);

    storageKey ||= pickFirstString([
      candidate?.storage_key,
      candidate?.storageKey,
      candidate?.key,
    ]);

    artifactName ||= pickFirstString([
      candidate?.original_artifact_name,
      candidate?.artifact_name,
      candidate?.artifactName,
      candidate?.filename,
      candidate?.name,
    ]);

    contentType ||= pickFirstString([
      candidate?.content_type,
      candidate?.contentType,
      candidate?.mime_type,
      candidate?.mimeType,
    ]);
  }

  return {
    downloadUrl,
    storageKey,
    artifactName,
    contentType,
  };
}

function extractResponseMessage(responseData, fallbackMessage = "") {
  return (
    pickFirstString([
      responseData?.detail?.message,
      responseData?.detail?.error,
      responseData?.message,
      responseData?.result?.message,
      responseData?.data?.message,
    ]) || fallbackMessage
  );
}

export default function ConvertPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t = convertPageTranslations[language] || convertPageTranslations.en;

  const downloadLabel = common.download || "Download output";
  const convertedFileLabel = t.convertedFile || "Converted file";
  const downloadReadyLabel = t.downloadReady || "Download ready";
  const outputReadyText =
    t.outputReadyText || "Your converted file is ready to download.";
  const missingDownloadUrlText =
    t.missingDownloadUrl ||
    "Conversion finished, but the backend did not return a download URL.";

  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [targetExtension, setTargetExtension] = useState("");
  const [conversionResult, setConversionResult] = useState("");
  const [downloadInfo, setDownloadInfo] = useState(null);

  const inputExtension = useMemo(() => {
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [selectedFile]);

  const allowedOutputs = useMemo(() => {
    return getAllowedOutputExtensions(inputExtension);
  }, [inputExtension]);

  const isValidFile = useMemo(() => {
    if (!selectedFile) return false;

    const ext = getFileExtension(selectedFile.name);
    const isAccepted = ACCEPTED_EXTENSIONS.includes(ext);
    const isWithinLimit = selectedFile.size <= MAX_FILE_SIZE_MB * 1024 * 1024;

    return isAccepted && isWithinLimit;
  }, [selectedFile]);

  const isValidConversion =
    !!inputExtension &&
    !!targetExtension &&
    allowedOutputs.includes(targetExtension);

  const canSubmit =
    !isSubmitting && !!selectedFile && isValidFile && isValidConversion;

  function resetResultState() {
    setConversionResult("");
    setDownloadInfo(null);
  }

  function rejectFile(message) {
    setSelectedFile(null);
    setTargetExtension("");
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

    const outputs = getAllowedOutputExtensions(ext);

    setError("");
    setSelectedFile(file);
    setTargetExtension(outputs[0] || "");
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

  function handleDownload() {
    if (!downloadInfo?.url) return;

    const link = document.createElement("a");
    link.href = downloadInfo.url;
    link.download = downloadInfo.filename || "converted-file";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError(t.chooseFileToConvert);
      return;
    }

    if (!isValidConversion) {
      setError(t.invalidConversion);
      return;
    }

    setIsSubmitting(true);
    setError("");
    resetResultState();

    try {
      const payload = {
        inputType: "file",
        filename: selectedFile.name,
        inputExtension,
        outputExtension: targetExtension,
      };

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("output_format", targetExtension.replace(".", ""));
      formData.append(
        "system_language",
        language === "fr" ? "french" : "english",
      );

      const response = await fetch("/api/analyzer/convert", {
        method: "POST",
        body: formData,
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.conversionFailed),
        );
      }

      const artifact = extractArtifactMetadata(responseData);
      const resolvedArtifactName =
        artifact.artifactName ||
        buildOutputFilename(selectedFile.name, payload.outputExtension);
      const backendMessage = extractResponseMessage(responseData);

      if (artifact.downloadUrl) {
        setDownloadInfo({
          url: artifact.downloadUrl,
          filename: resolvedArtifactName,
          contentType: artifact.contentType,
          storageKey: artifact.storageKey,
        });
      }

      const previewLines = [
        t.conversionCompleted,
        "",
        `${t.inputFile}: ${selectedFile.name}`,
        `${t.inputExtension}: ${payload.inputExtension}`,
        `${t.outputExtension}: ${payload.outputExtension}`,
        `${convertedFileLabel}: ${resolvedArtifactName}`,
      ];

      if (artifact.downloadUrl) {
        previewLines.push("", outputReadyText);
      } else {
        previewLines.push("", missingDownloadUrlText);
      }

      if (backendMessage) {
        previewLines.push("", backendMessage);
      } else {
        previewLines.push("", t.conversionMatchesRules);
      }

      setConversionResult(previewLines.join("\n"));
    } catch (submitError) {
      setError(submitError?.message || t.conversionFailed);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <div className="relative isolate overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_28%),radial-gradient(circle_at_top_right,rgba(168,85,247,0.18),transparent_30%),linear-gradient(to_bottom,#081120,#0a1426,#07111f)]" />

        <div className="relative mx-auto max-w-5xl px-6 py-12 md:px-8 md:py-16">
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

          <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <form
              onSubmit={handleSubmit}
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
                    {t.allowedInputs}{" "}
                    <span className="font-medium text-white">
                      .pdf, .docx, .jpg, .jpeg, .png
                    </span>
                  </p>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.jpg,.jpeg,.png"
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
                          {common.fileAccepted}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {selectedFile.name} • {formatBytes(selectedFile.size)}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {t.detectedType}{" "}
                          {getInputTypeLabel(inputExtension, t)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {selectedFile && isValidFile && (
                  <div className="mt-6 grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="block">
                        <span className="mb-3 block text-sm font-medium text-white/80">
                          {t.from}
                        </span>
                        <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/85">
                          <FileType className="h-4 w-4 text-cyan-300" />
                          {inputExtension}
                        </div>
                      </label>
                    </div>

                    <div>
                      <label className="block">
                        <span className="mb-3 block text-sm font-medium text-white/80">
                          {t.convertTo}
                        </span>
                        <select
                          value={targetExtension}
                          onChange={(e) => {
                            setTargetExtension(e.target.value);
                            setError("");
                            resetResultState();
                          }}
                          className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-300/40 focus:bg-white/10"
                        >
                          {allowedOutputs.map((ext) => (
                            <option
                              key={ext}
                              value={ext}
                              className="bg-slate-900 text-white"
                            >
                              {ext}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                )}

                {selectedFile && isValidFile && (
                  <div className="mt-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-4">
                    <div className="flex items-start gap-3">
                      <Repeat className="mt-0.5 h-5 w-5 text-cyan-300" />
                      <div className="text-sm leading-6 text-cyan-100">
                        {t.allowedOutputsFor}{" "}
                        <span className="font-semibold">{inputExtension}</span>:{" "}
                        {allowedOutputs.length > 0
                          ? allowedOutputs.join(", ")
                          : t.none}
                      </div>
                    </div>
                  </div>
                )}

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
                    disabled={!canSubmit}
                    className={`rounded-2xl px-5 py-3 text-sm font-semibold transition ${
                      canSubmit
                        ? "bg-white text-slate-900 hover:scale-[1.02] hover:shadow-xl"
                        : "cursor-not-allowed bg-white/10 text-white/40"
                    }`}
                  >
                    {isSubmitting ? common.converting : common.convert}
                  </button>

                  {downloadInfo?.url && (
                    <button
                      type="button"
                      onClick={handleDownload}
                      className="inline-flex items-center gap-2 rounded-2xl border border-cyan-300/30 bg-cyan-400/10 px-5 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/15"
                    >
                      <Download className="h-4 w-4" />
                      {downloadLabel}
                    </button>
                  )}

                  <div className="text-sm text-white/55">
                    {t.conversionLabel}{" "}
                    <span className="font-medium text-white/85">
                      {inputExtension || "—"} → {targetExtension || "—"}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl lg:sticky lg:top-6">
                <h2 className="text-lg font-semibold text-white">
                  {t.conversionOutput}
                </h2>
                <p className="mt-1 text-sm text-white/55">
                  {common.previewArea}
                </p>

                <div className="mt-4 rounded-2xl border border-white/10 bg-[#081120] p-4">
                  {conversionResult ? (
                    <div className="space-y-4">
                      <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                        {conversionResult}
                      </pre>

                      {downloadInfo?.url && (
                        <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                          <div className="flex items-start gap-3">
                            <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
                            <div>
                              <p className="font-medium text-emerald-100">
                                {downloadReadyLabel}
                              </p>
                              <p className="mt-1 text-sm text-emerald-100/80">
                                {downloadInfo.filename}
                              </p>
                              <button
                                type="button"
                                onClick={handleDownload}
                                className="mt-3 inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
                              >
                                <Download className="h-4 w-4" />
                                {downloadLabel}
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm leading-6 text-white/45">
                      {t.previewText}
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <RefreshCw className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {t.allowedConversions}
                    </h2>
                    <p className="text-sm text-white/55">
                      {t.strictConversionMatrix}
                    </p>
                  </div>
                </div>

                <div className="space-y-3 text-sm leading-6 text-white/70">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">{t.pdfWordTitle}</p>
                    <p className="mt-1 text-white/65">{t.pdfWordDescription}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.jpgWordPdfTitle}
                    </p>
                    <p className="mt-1 text-white/65">
                      {t.jpgWordPdfDescription}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">{t.pngJpgTitle}</p>
                    <p className="mt-1 text-white/65">{t.pngJpgDescription}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <h2 className="text-lg font-semibold text-white">
                  {t.inputCoverage}
                </h2>
                <p className="mt-1 text-sm text-white/55">
                  {t.supportedUploadTypes}
                </p>

                <div className="mt-4 space-y-3 text-sm leading-6 text-white/70">
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <FileText className="h-4 w-4 text-cyan-300" />
                    <span>.pdf</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <FileText className="h-4 w-4 text-cyan-300" />
                    <span>.docx</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <ImageIcon className="h-4 w-4 text-cyan-300" />
                    <span>.jpg / .jpeg</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <ImageIcon className="h-4 w-4 text-cyan-300" />
                    <span>.png</span>
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
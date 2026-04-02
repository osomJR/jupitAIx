"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/components/language_provider";
import {
  ArrowLeft,
  Upload,
  Sparkles,
  XCircle,
  CheckCircle2,
  FileText,
  AlignLeft,
  ShieldCheck,
  BookOpen,
} from "lucide-react";
import {
  commonTranslations,
  explainPageTranslations,
} from "@/lib/translations";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];
const REJECTED_EXTENSIONS = [".png", ".jpg", ".jpeg"];
const MAX_FILE_SIZE_MB = 10;
const INLINE_TEXT_EXTENSION = ".txt";

function getFileExtension(filename = "") {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
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

export default function ExplainPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t = explainPageTranslations[language] || explainPageTranslations.en;

  const [mode, setMode] = useState("file");
  const [selectedFile, setSelectedFile] = useState(null);
  const [inlineText, setInlineText] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [explanationResult, setExplanationResult] = useState("");
  const [downloadInfo, setDownloadInfo] = useState(null);

  const inputExtension = useMemo(() => {
    if (mode === "text") return INLINE_TEXT_EXTENSION;
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [mode, selectedFile]);

  const outputExtension =
    inputExtension || (mode === "text" ? INLINE_TEXT_EXTENSION : "");

  const isValidFile = useMemo(() => {
    if (!selectedFile) return false;
    const ext = getFileExtension(selectedFile.name);
    const isAccepted = ACCEPTED_EXTENSIONS.includes(ext);
    const isWithinLimit = selectedFile.size <= MAX_FILE_SIZE_MB * 1024 * 1024;
    return isAccepted && isWithinLimit;
  }, [selectedFile]);

  const canSubmit =
    !isSubmitting &&
    ((mode === "file" && selectedFile && isValidFile) ||
      (mode === "text" && inlineText.trim().length > 0));

  function resetResultState() {
    setExplanationResult("");
    setDownloadInfo(null);
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

  async function handleSubmit(event) {
    event.preventDefault();

    if (mode === "file" && !selectedFile) {
      setError(replaceVars(t.unsupportedFileType, { ext: "unknown" }));
      return;
    }

    if (mode === "text" && !inlineText.trim()) {
      setError(t.explanationFailed);
      return;
    }

    setIsSubmitting(true);
    setError("");
    resetResultState();

    try {
      const formData = new FormData();

      if (mode === "file") {
        formData.append("file", selectedFile);
      } else {
        formData.append("text", inlineText.trim());
      }

      formData.append(
        "system_language",
        language === "fr" ? "french" : "english",
      );
      formData.append("allow_external_knowledge", "false");

      const backendBase = process.env.NEXT_PUBLIC_API_BASE_URL;
      if (!backendBase) {
        throw new Error("NEXT_PUBLIC_API_BASE_URL is not set.");
      }
      const res = await fetch(`${backendBase}/api/v1/analyzer/explain`, {
        method: "POST",
        body: formData,
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        data = null;
      }

      if (!res.ok) {
        throw new Error(
          data?.detail?.message ||
            data?.detail ||
            data?.message ||
            "Explain request failed.",
        );
      }

      const result = data?.result;

      if (result?.content) {
        setExplanationResult(result.content);
        setDownloadInfo(null);
      } else if (
        result?.filename ||
        result?.storage_key ||
        result?.download_url
      ) {
        setExplanationResult("");
        setDownloadInfo({
          filename: result.filename,
          outputFormat: result.output_format,
          fileSizeMb: result.file_size_mb,
          url:
            result.download_url ||
            `${backendBase}/api/v1/analyzer/artifacts/${result.storage_key}`,
        });
      } else {
        throw new Error("Unexpected response shape from backend.");
      }
    } catch (err) {
      setError(err.message || t.explanationFailed);
    } finally {
      setIsSubmitting(false);
    }
  }
  return (
    <main className="min-h-screen bg-[#07111f] text-white">
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
                <div className="mb-6 inline-flex rounded-2xl border border-white/10 bg-white/5 p-1">
                  <button
                    type="button"
                    onClick={() => {
                      setMode("file");
                      setError("");
                      resetResultState();
                    }}
                    className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                      mode === "file"
                        ? "bg-white text-slate-900 shadow-sm"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                  >
                    {t.fileMode}
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setMode("text");
                      setSelectedFile(null);
                      setError("");
                      resetResultState();
                    }}
                    className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                      mode === "text"
                        ? "bg-white text-slate-900 shadow-sm"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                  >
                    {t.textMode}
                  </button>
                </div>

                {mode === "file" ? (
                  <>
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
                          {inputExtension || ".pdf / .docx"}
                        </span>
                      </p>

                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.docx"
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
                              {selectedFile.name} •{" "}
                              {formatBytes(selectedFile.size)}
                            </p>
                            <p className="mt-1 text-sm text-emerald-100/80">
                              {t.outputFormatLabel} {outputExtension}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="rounded-3xl border border-white/15 bg-white/5 p-5">
                    <label className="block">
                      <span className="mb-3 block text-sm font-medium text-white/80">
                        {t.pasteTextLabel}
                      </span>
                      <textarea
                        value={inlineText}
                        onChange={(e) => {
                          setInlineText(e.target.value);
                          setError("");
                          resetResultState();
                        }}
                        placeholder={t.pasteTextPlaceholder}
                        rows={10}
                        className="w-full rounded-2xl border border-white/10 bg-[#081120] px-4 py-3 text-sm leading-6 text-white outline-none transition placeholder:text-white/30 focus:border-cyan-300/40"
                      />
                    </label>

                    <p className="mt-3 text-sm leading-6 text-white/55">
                      {t.inlineTextTreatedAs}
                    </p>
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
                    {isSubmitting ? common.explaining : common.explain}
                  </button>

                  <div className="text-sm text-white/55">
                    {common.outputFormat}{" "}
                    <span className="font-medium text-white/85">
                      {outputExtension || "—"}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="space-y-6">
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
                    <p className="mt-1 text-white/65">.pdf, .docx</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.inlineInputLabel}
                    </p>
                    <p className="mt-1 text-white/65">{t.inlineInputValue}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.rejectedAutomaticallyLabel}
                    </p>
                    <p className="mt-1 text-white/65">
                      {t.rejectedAutomaticallyValue}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.outputRuleLabel}
                    </p>
                    <p className="mt-1 text-white/65">{t.outputRuleValue}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <FileText className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {t.explanationOutputTitle}
                    </h2>
                    <p className="text-sm text-white/55">
                      {common.previewArea}
                    </p>
                  </div>
                </div>

                {explanationResult && (
                  <div className="rounded-2xl border border-white/10 bg-[#081120] p-4">
                    <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                      {explanationResult}
                    </pre>
                  </div>
                )}

                {downloadInfo && (
                  <div className="rounded-2xl border border-white/10 bg-[#081120] p-4">
                    <div className="space-y-2 text-sm text-white/80">
                      <p>
                        <span className="font-medium text-white">File:</span>{" "}
                        {downloadInfo.filename}
                      </p>
                      <p>
                        <span className="font-medium text-white">Format:</span>{" "}
                        {downloadInfo.outputFormat}
                      </p>
                      <p>
                        <span className="font-medium text-white">Size:</span>{" "}
                        {downloadInfo.fileSizeMb} MB
                      </p>
                    </div>

                    <a
                      href={downloadInfo.url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-5 inline-flex rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
                    >
                      Download explained file
                    </a>
                  </div>
                )}
                {!explanationResult && !downloadInfo && (
                  <div className="rounded-2xl border border-white/10 bg-[#081120] p-4">
                    <p className="text-sm leading-6 text-white/45">
                      {t.previewEmpty}
                    </p>
                  </div>
                )}

                <div className="mt-4 inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/75">
                  <BookOpen className="h-4 w-4 text-cyan-300" />
                  {t.outputExtensionLabel}{" "}
                  <span className="font-medium text-white">
                    {downloadInfo?.outputFormat || outputExtension || "—"}
                  </span>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <h2 className="text-lg font-semibold text-white">
                  {t.policyTitle}
                </h2>
                <p className="mt-1 text-sm text-white/55">{t.policySubtitle}</p>

                <div className="mt-4 space-y-3 text-sm leading-6 text-white/70">
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <FileText className="h-4 w-4 text-cyan-300" />
                    <span>.pdf / .docx</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <AlignLeft className="h-4 w-4 text-cyan-300" />
                    <span>{t.inlineInputValue}</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                    <XCircle className="h-4 w-4 text-cyan-300" />
                    <span>{REJECTED_EXTENSIONS.join(", ")}</span>
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
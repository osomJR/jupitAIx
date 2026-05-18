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
  Languages,
  ShieldCheck,
  Globe2,
  AlignLeft,
} from "lucide-react";
import {
  commonTranslations,
  translatePageTranslations,
} from "@/lib/translations";
import AppSidebarLayout from "@/components/app_sidebar";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];
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

export default function TranslatePage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t = translatePageTranslations[language] || translatePageTranslations.en;

  const [mode, setMode] = useState("file");
  const [selectedFile, setSelectedFile] = useState(null);
  const [inlineText, setInlineText] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [translationResult, setTranslationResult] = useState("");

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
    targetLanguage.trim().length > 0 &&
    ((mode === "file" && selectedFile && isValidFile) ||
      (mode === "text" && inlineText.trim().length > 0));

  function resetResultState() {
    setTranslationResult("");
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

    if (!targetLanguage.trim()) {
      setError(t.targetLanguageRequired);
      return;
    }

    if (mode === "file" && !selectedFile) {
      setError(
        replaceVars(t.unsupportedFileType, {
          ext: "unknown",
        }),
      );
      return;
    }

    if (mode === "text" && !inlineText.trim()) {
      setError(t.translationFailed);
      return;
    }

    setIsSubmitting(true);
    setError("");
    resetResultState();

    try {
      await new Promise((resolve) => setTimeout(resolve, 900));

      if (mode === "text") {
        const mockResult = `${t.inlineTranslationIntro}

${t.targetLanguageResultLabel} ${targetLanguage}
${t.outputExtensionResultLabel} ${INLINE_TEXT_EXTENSION}

${t.translatedPreviewLabel}
[${targetLanguage}] ${inlineText.slice(0, 180)}${
          inlineText.length > 180 ? "..." : ""
        }`;

        setTranslationResult(mockResult);
      } else {
        const mockResult = `${replaceVars(t.fileTranslationIntro, {
          filename: selectedFile.name,
        })}

${t.targetLanguageResultLabel} ${targetLanguage}
${t.inputExtensionLabel} ${inputExtension}
${t.outputExtensionResultLabel} ${outputExtension}

${t.preservedExtensionMessage}

${t.translatedPreviewLabel}
[${targetLanguage}] Document content translated while preserving file format compatibility.`;

        setTranslationResult(mockResult);
      }
    } catch (submitError) {
      setError(t.translationFailed);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AppSidebarLayout>
      <div className="relative isolate min-h-screen overflow-hidden bg-[var(--app-bg)] text-[var(--app-text)]">
        <div className="absolute inset-0 bg-[var(--app-bg)]" />

        <div className="relative mx-auto max-w-5xl px-6 py-12 md:px-8 md:py-16">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="mb-8 inline-flex items-center gap-2 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] px-4 py-2 text-sm app-text-muted backdrop-blur transition hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
          >
            <ArrowLeft className="h-4 w-4" />
            {common.back}
          </button>

          <section className="mb-10">
            <div className="inline-flex items-center gap-2 rounded-full border border-[var(--app-accent-border)] bg-[var(--app-accent-bg)] px-4 py-2 text-sm text-[var(--app-accent-text)] backdrop-blur">
              <Sparkles className="h-4 w-4" />
              {t.badge}
            </div>

            <div className="mt-6 max-w-3xl">
              <h1 className="text-4xl font-semibold tracking-tight text-[var(--app-text)] sm:text-5xl">
                {t.title}
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 app-text-muted md:text-lg">
                {t.description}
              </p>
            </div>
          </section>

          <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <form
              onSubmit={handleSubmit}
              className="relative overflow-hidden rounded-3xl border border-[var(--app-border)] bg-[var(--app-surface-strong)] p-6 backdrop-blur-xl md:p-8"
            >
              <div className="absolute inset-0 app-card-overlay" />

              <div className="relative">
                <div className="mb-6 inline-flex rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-1">
                  <button
                    type="button"
                    onClick={() => {
                      setMode("file");
                      setError("");
                      resetResultState();
                    }}
                    className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                      mode === "file"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)] shadow-sm"
                        : "app-text-muted hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
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
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)] shadow-sm"
                        : "app-text-muted hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
                    }`}
                  >
                    {t.textMode}
                  </button>
                </div>

                <div className="mb-6">
                  <label className="block">
                    <span className="mb-3 block text-sm font-medium app-text-muted">
                      {t.targetLanguageLabel}
                    </span>
                    <input
                      type="text"
                      value={targetLanguage}
                      onChange={(e) => {
                        setTargetLanguage(e.target.value);
                        setError("");
                      }}
                      placeholder={t.targetLanguagePlaceholder}
                      list="translation-language-suggestions"
                      className="w-full rounded-2xl border border-[var(--app-border)] bg-[var(--app-panel)] px-4 py-3 text-sm text-[var(--app-text)] outline-none transition placeholder:text-[var(--app-text-soft)] focus:border-[var(--app-accent-border)]"
                    />
                    <datalist id="translation-language-suggestions">
                      {t.languageSuggestions.map((item) => (
                        <option key={item} value={item} />
                      ))}
                    </datalist>
                  </label>

                  <p className="mt-3 text-sm leading-6 app-text-soft">
                    {t.targetLanguageHelp}
                  </p>
                </div>

                {mode === "file" ? (
                  <>
                    <div
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      className="rounded-3xl border border-dashed border-[var(--app-border)] bg-[var(--app-surface)] p-8 text-center transition hover:border-[var(--app-border-strong)] hover:bg-[var(--app-surface-strong)]"
                    >
                      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)]">
                        <Upload className="h-7 w-7 text-cyan-300" />
                      </div>

                      <h2 className="text-lg font-semibold text-[var(--app-text)]">
                        {t.uploadTitle}
                      </h2>
                      <p className="mt-2 text-sm leading-6 app-text-muted">
                        {t.allowedFileInputs}
                      </p>
                      <p className="mt-2 text-sm leading-6 app-text-soft">
                        {t.outputExtensionWillBe}{" "}
                        <span className="font-medium text-[var(--app-text)]">
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
                        className="mt-5 rounded-2xl bg-[var(--app-button-bg)] px-5 py-3 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.02] hover:shadow-xl"
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
                              {common.outputFormat} {outputExtension}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="rounded-3xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
                    <label className="block">
                      <span className="mb-3 block text-sm font-medium app-text-muted">
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
                        className="w-full rounded-2xl border border-[var(--app-border)] bg-[var(--app-panel)] px-4 py-3 text-sm leading-6 text-[var(--app-text)] outline-none transition placeholder:text-[var(--app-text-soft)] focus:border-[var(--app-accent-border)]"
                      />
                    </label>

                    <p className="mt-3 text-sm leading-6 app-text-soft">
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
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)] hover:scale-[1.02] hover:shadow-xl"
                        : "cursor-not-allowed bg-[var(--app-surface)] app-text-soft"
                    }`}
                  >
                    {isSubmitting ? common.translating : common.translate}
                  </button>

                  <div className="text-sm app-text-soft">
                    {common.outputFormat}{" "}
                    <span className="font-medium app-text-muted">
                      {outputExtension || "—"}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-[var(--app-border)] bg-[var(--app-surface-strong)] p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)]">
                    <ShieldCheck className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-[var(--app-text)]">
                      {common.formatPolicy}
                    </h2>
                    <p className="text-sm app-text-soft">{t.policySubtitle}</p>
                  </div>
                </div>

                <div className="space-y-3 text-sm leading-6 app-text-muted">
                  <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="font-semibold text-[var(--app-text)]">
                      {t.allowedUploadsLabel}
                    </p>
                    <p className="mt-1 app-text-muted">.pdf, .docx</p>
                  </div>

                  <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="font-semibold text-[var(--app-text)]">
                      {t.inlineInputLabel}
                    </p>
                    <p className="mt-1 app-text-muted">{t.inlineInputValue}</p>
                  </div>

                  <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="font-semibold text-[var(--app-text)]">
                      {t.rejectedAutomaticallyLabel}
                    </p>
                    <p className="mt-1 app-text-muted">
                      {t.rejectedAutomaticallyValue}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="font-semibold text-[var(--app-text)]">
                      {t.outputRuleLabel}
                    </p>
                    <p className="mt-1 app-text-muted">{t.outputRuleValue}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-[var(--app-border)] bg-[var(--app-surface-strong)] p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)]">
                    <Globe2 className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-[var(--app-text)]">
                      {t.translationOutputTitle}
                    </h2>
                    <p className="text-sm app-text-soft">
                      {common.previewArea}
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-panel)] p-4">
                  {translationResult ? (
                    <pre className="whitespace-pre-wrap break-words text-sm leading-7 app-text-muted">
                      {translationResult}
                    </pre>
                  ) : (
                    <p className="text-sm leading-6 app-text-soft">
                      {t.previewEmpty}
                    </p>
                  )}
                </div>

                <div className="mt-4 inline-flex items-center gap-2 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] px-4 py-3 text-sm app-text-muted">
                  <Languages className="h-4 w-4 text-cyan-300" />
                  {t.outputExtensionLabel}{" "}
                  <span className="font-medium text-[var(--app-text)]">
                    {outputExtension || "—"}
                  </span>
                </div>
              </div>

              <div className="rounded-3xl border border-[var(--app-border)] bg-[var(--app-surface-strong)] p-6 backdrop-blur-xl">
                <h2 className="text-lg font-semibold text-[var(--app-text)]">
                  {common.formatPolicy}
                </h2>
                <p className="mt-1 text-sm app-text-soft">{t.policySubtitle}</p>

                <div className="mt-4 space-y-3 text-sm leading-6 app-text-muted">
                  <div className="flex items-center gap-3 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <FileText className="h-4 w-4 text-cyan-300" />
                    <span>.pdf / .docx</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <AlignLeft className="h-4 w-4 text-cyan-300" />
                    <span>{t.inlineInputValue}</span>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <XCircle className="h-4 w-4 text-cyan-300" />
                    <span>.png, .jpg, .jpeg</span>
                  </div>
                </div>
              </div>
            </aside>
          </section>
        </div>
      </div>
    </AppSidebarLayout>
  );
}
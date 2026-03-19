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

export default function ConvertPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t = convertPageTranslations[language] || convertPageTranslations.en;

  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [targetExtension, setTargetExtension] = useState("");
  const [conversionResult, setConversionResult] = useState("");

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
    setConversionResult("");
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

      // Replace with your real backend call.
      // Example:
      // const formData = new FormData();
      // formData.append("file", selectedFile);
      // formData.append("inputExtension", payload.inputExtension);
      // formData.append("outputExtension", payload.outputExtension);
      // await fetch("/api/convert", { method: "POST", body: formData });

      await new Promise((resolve) => setTimeout(resolve, 900));

      const mockResult = `${t.conversionCompleted}

${t.inputFile}: ${selectedFile.name}
${t.inputExtension}: ${payload.inputExtension}
${t.outputExtension}: ${payload.outputExtension}

${t.conversionMatchesRules}`;

      setConversionResult(mockResult);
    } catch (submitError) {
      setError(t.conversionFailed);
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

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <h2 className="text-lg font-semibold text-white">
                  {t.conversionOutput}
                </h2>
                <p className="mt-1 text-sm text-white/55">
                  {common.previewArea}
                </p>

                <div className="mt-4 rounded-2xl border border-white/10 bg-[#081120] p-4">
                  {conversionResult ? (
                    <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                      {conversionResult}
                    </pre>
                  ) : (
                    <p className="text-sm leading-6 text-white/45">
                      {t.previewText}
                    </p>
                  )}
                </div>
              </div>
            </aside>
          </section>
        </div>
      </div>
    </main>
  );
}

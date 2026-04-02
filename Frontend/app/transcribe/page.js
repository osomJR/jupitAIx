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
  AlignLeft,
  ShieldCheck,
  Mic,
} from "lucide-react";
import {
  commonTranslations,
  transcribePageTranslations,
} from "@/lib/translations";

const ACCEPTED_EXTENSIONS = [".mp3", ".mp4", ".mkv", ".mov"];
const AUDIO_EXTENSIONS = [".mp3"];
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"];

const OUTPUT_EXTENSION = ".txt";
const MAX_AUDIO_FILE_SIZE_MB = 10;
const MAX_VIDEO_FILE_SIZE_MB = 25;
const MAX_AUDIO_DURATION_SECONDS = 120;
const MAX_VIDEO_DURATION_SECONDS = 180;

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

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const wholeSeconds = Math.round(seconds);
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function replaceVars(template, vars = {}) {
  return template.replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? "");
}

function getMediaType(extension = "") {
  if (AUDIO_EXTENSIONS.includes(extension)) return "audio";
  if (VIDEO_EXTENSIONS.includes(extension)) return "video";
  return "unknown";
}

function getMaxFileSizeMb(mediaType) {
  return mediaType === "audio"
    ? MAX_AUDIO_FILE_SIZE_MB
    : MAX_VIDEO_FILE_SIZE_MB;
}

function getMaxDurationSeconds(mediaType) {
  return mediaType === "audio"
    ? MAX_AUDIO_DURATION_SECONDS
    : MAX_VIDEO_DURATION_SECONDS;
}

function getMediaTypeLabel(extension, t) {
  const mediaType = getMediaType(extension);
  if (mediaType === "audio") return t.audioType;
  if (mediaType === "video") return t.videoType;
  return t.unknownType;
}

function pickFirstString(values = []) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
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
  if (
    typeof responseData?.message === "string" &&
    responseData.message.trim()
  ) {
    return responseData.message.trim();
  }
  if (typeof responseData?.error === "string" && responseData.error.trim()) {
    return responseData.error.trim();
  }

  try {
    return JSON.stringify(detail || responseData);
  } catch {
    return fallbackMessage;
  }
}

function extractTranscriptText(responseData) {
  const candidates = [
    responseData?.result,
    responseData?.data,
    responseData,
  ].filter(Boolean);

  for (const candidate of candidates) {
    const directText = pickFirstString([
      candidate?.content,
      candidate?.transcript_text,
      candidate?.transcriptText,
      candidate?.transcript,
      candidate?.text,
    ]);

    if (directText) {
      return directText;
    }

    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  return "";
}

function readMediaDuration(file, mediaType) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const element =
      mediaType === "audio"
        ? document.createElement("audio")
        : document.createElement("video");

    let settled = false;

    const cleanup = () => {
      element.removeAttribute("src");
      element.load();
      URL.revokeObjectURL(objectUrl);
    };

    element.preload = "metadata";

    element.onloadedmetadata = () => {
      if (settled) return;
      settled = true;

      const duration = Number(element.duration);
      cleanup();

      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error("Could not read media duration."));
        return;
      }

      resolve(duration);
    };

    element.onerror = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Could not read media duration."));
    };

    element.src = objectUrl;
  });
}

async function validatePickedFile(file, t) {
  const extension = getFileExtension(file.name);

  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    throw new Error(
      replaceVars(t.unsupportedFileType, {
        ext: extension || "unknown",
      }),
    );
  }

  const mediaType = getMediaType(extension);
  if (mediaType === "unknown") {
    throw new Error(
      replaceVars(t.unsupportedFileType, {
        ext: extension || "unknown",
      }),
    );
  }

  const maxSizeMb = getMaxFileSizeMb(mediaType);
  if (file.size > maxSizeMb * 1024 * 1024) {
    throw new Error(
      replaceVars(t.fileTooLarge, {
        maxSize: maxSizeMb,
      }),
    );
  }

  let durationSeconds;
  try {
    durationSeconds = await readMediaDuration(file, mediaType);
  } catch {
    throw new Error(t.couldNotReadDuration);
  }

  const maxDurationSeconds = getMaxDurationSeconds(mediaType);
  if (durationSeconds > maxDurationSeconds) {
    throw new Error(
      replaceVars(t.mediaTooLong, {
        maxDuration: formatDuration(maxDurationSeconds),
      }),
    );
  }

  return {
    mediaType,
    durationSeconds,
    maxSizeMb,
    maxDurationSeconds,
  };
}

export default function TranscribePage() {
  const router = useRouter();
  const fileInputRef = useRef(null);
  const { language } = useLanguage();

  const common = commonTranslations[language] || commonTranslations.en;
  const t =
    transcribePageTranslations[language] || transcribePageTranslations.en;

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileMeta, setSelectedFileMeta] = useState(null);
  const [error, setError] = useState("");
  const [isCheckingFile, setIsCheckingFile] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [transcriptResult, setTranscriptResult] = useState("");
  const [preserveFillerWords, setPreserveFillerWords] = useState(true);
  const [removeBackgroundNoise, setRemoveBackgroundNoise] = useState(false);
  const [diarizeSpeakers, setDiarizeSpeakers] = useState(true);

  const inputExtension = useMemo(() => {
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [selectedFile]);

  const canSubmit =
    !isCheckingFile && !isSubmitting && !!selectedFile && !!selectedFileMeta;

  function resetResultState() {
    setTranscriptResult("");
  }

  function resetFileState() {
    setSelectedFile(null);
    setSelectedFileMeta(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function rejectFile(message) {
    resetFileState();
    setError(message);
    resetResultState();
  }

  async function handlePickedFile(file) {
    if (!file) return;

    setIsCheckingFile(true);
    setError("");
    resetResultState();

    try {
      const validated = await validatePickedFile(file, t);
      setSelectedFile(file);
      setSelectedFileMeta(validated);
    } catch (pickedFileError) {
      rejectFile(pickedFileError?.message || t.unsupportedFileType);
    } finally {
      setIsCheckingFile(false);
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    void handlePickedFile(file);
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
    void handlePickedFile(file);
  }

  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile || !selectedFileMeta) {
      setError(t.chooseFileToTranscribe);
      return;
    }

    setIsSubmitting(true);
    setError("");
    resetResultState();

    try {
      const extension = getFileExtension(selectedFile.name);

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("media_type", selectedFileMeta.mediaType);
      formData.append("media_format", extension.replace(".", ""));
      formData.append(
        "duration_seconds",
        String(Math.round(selectedFileMeta.durationSeconds)),
      );
      formData.append(
        "system_language",
        language === "fr" ? "french" : "english",
      );
      formData.append("preserve_filler_words", String(preserveFillerWords));
      formData.append("remove_background_noise", String(removeBackgroundNoise));
      formData.append("diarize_speakers", String(diarizeSpeakers));

      const response = await fetch("/api/analyzer/transcribe", {
        method: "POST",
        body: formData,
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractResponseMessage(responseData, t.transcriptionFailed),
        );
      }

      const transcriptText = extractTranscriptText(responseData);

      if (!transcriptText) {
        throw new Error("Backend returned no transcript text.");
      }

      setTranscriptResult(transcriptText);
    } catch (submitError) {
      setError(submitError?.message || t.transcriptionFailed);
    } finally {
      setIsSubmitting(false);
    }
  }

  const actionLabel = isCheckingFile
    ? t.validatingMedia
    : isSubmitting
      ? common.transcribing || common.generating
      : common.transcribe;

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
                    {t.allowedFileInputs}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/55">
                    {t.outputExtensionWillBe}{" "}
                    <span className="font-medium text-white">
                      {OUTPUT_EXTENSION}
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

                {selectedFile && selectedFileMeta && (
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
                          {t.detectedTypeLabel}{" "}
                          {getMediaTypeLabel(inputExtension, t)}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {t.durationLabel}{" "}
                          {formatDuration(selectedFileMeta.durationSeconds)}
                        </p>
                        <p className="mt-1 text-sm text-emerald-100/80">
                          {common.outputFormat} {OUTPUT_EXTENSION}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-6 rounded-3xl border border-white/15 bg-white/5 p-5">
                  <div className="mb-4 flex items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                      <Mic className="h-5 w-5 text-cyan-300" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">
                        {t.transcriptOptionsTitle}
                      </h2>
                      <p className="text-sm text-white/55">
                        {t.transcriptOptionsSubtitle}
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <label className="flex items-start justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div>
                        <p className="text-sm font-semibold text-white">
                          {t.preserveFillerWordsLabel}
                        </p>
                        <p className="mt-1 text-sm leading-6 text-white/60">
                          {t.preserveFillerWordsHelp}
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={preserveFillerWords}
                        onChange={(e) =>
                          setPreserveFillerWords(e.target.checked)
                        }
                        className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent text-cyan-300 focus:ring-cyan-300"
                      />
                    </label>

                    <label className="flex items-start justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div>
                        <p className="text-sm font-semibold text-white">
                          {t.removeBackgroundNoiseLabel}
                        </p>
                        <p className="mt-1 text-sm leading-6 text-white/60">
                          {t.removeBackgroundNoiseHelp}
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={removeBackgroundNoise}
                        onChange={(e) =>
                          setRemoveBackgroundNoise(e.target.checked)
                        }
                        className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent text-cyan-300 focus:ring-cyan-300"
                      />
                    </label>

                    <label className="flex items-start justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div>
                        <p className="text-sm font-semibold text-white">
                          {t.diarizeSpeakersLabel}
                        </p>
                        <p className="mt-1 text-sm leading-6 text-white/60">
                          {t.diarizeSpeakersHelp}
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={diarizeSpeakers}
                        onChange={(e) => setDiarizeSpeakers(e.target.checked)}
                        className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent text-cyan-300 focus:ring-cyan-300"
                      />
                    </label>
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
                    disabled={!canSubmit}
                    className={`rounded-2xl px-5 py-3 text-sm font-semibold transition ${
                      canSubmit
                        ? "bg-white text-slate-900 hover:scale-[1.02] hover:shadow-xl"
                        : "cursor-not-allowed bg-white/10 text-white/40"
                    }`}
                  >
                    {actionLabel}
                  </button>

                  <div className="text-sm text-white/55">
                    {common.outputFormat}{" "}
                    <span className="font-medium text-white/85">
                      {OUTPUT_EXTENSION}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl lg:sticky lg:top-6">
                <h2 className="text-lg font-semibold text-white">
                  {t.transcriptOutput}
                </h2>
                <p className="mt-1 text-sm text-white/55">
                  {common.previewArea}
                </p>

                <div className="mt-4 rounded-2xl border border-white/10 bg-[#081120] p-4">
                  {transcriptResult ? (
                    <div className="space-y-4">
                      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                        <div className="flex items-start gap-3">
                          <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
                          <div>
                            <p className="font-medium text-emerald-100">
                              {t.transcriptReady}
                            </p>
                            <p className="mt-1 text-sm text-emerald-100/80">
                              {t.transcriptReadyText}
                            </p>
                            <p className="mt-1 text-sm text-emerald-100/80">
                              {t.transcriptMetaLabel}: {t.transcriptMetaValue}
                            </p>
                          </div>
                        </div>
                      </div>

                      <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                        {transcriptResult}
                      </pre>
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
                    <p className="font-semibold text-white">{t.limitsLabel}</p>
                    <p className="mt-1 text-white/65">{t.limitsValueAudio}</p>
                    <p className="mt-1 text-white/65">{t.limitsValueVideo}</p>
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
                    <AlignLeft className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {t.processingRulesTitle}
                    </h2>
                    <p className="text-sm text-white/55">
                      {t.processingRulesSubtitle}
                    </p>
                  </div>
                </div>

                <div className="space-y-3 text-sm leading-6 text-white/70">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.preserveLanguageTitle}
                    </p>
                    <p className="mt-1 text-white/65">
                      {t.preserveLanguageDescription}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.noRewriteTitle}
                    </p>
                    <p className="mt-1 text-white/65">
                      {t.noRewriteDescription}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="font-semibold text-white">
                      {t.videoHandlingTitle}
                    </p>
                    <p className="mt-1 text-white/65">
                      {t.videoHandlingDescription}
                    </p>
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

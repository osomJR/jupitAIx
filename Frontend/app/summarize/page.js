"use client";
import { useLanguage } from "@/components/language_provider";
import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  FileText,
  Upload,
  Sparkles,
  XCircle,
  CheckCircle2,
  FileType,
  Type,
} from "lucide-react";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];
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

export default function SummarizePage() {
  const router = useRouter();
  const fileInputRef = useRef(null);

  const [mode, setMode] = useState("file"); // "file" | "text"
  const [selectedFile, setSelectedFile] = useState(null);
  const [inlineText, setInlineText] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [summaryResult, setSummaryResult] = useState("");
  const [outputExtension, setOutputExtension] = useState("");

  const derivedInputExtension = useMemo(() => {
    if (mode === "text") return ".txt";
    if (!selectedFile) return "";
    return getFileExtension(selectedFile.name);
  }, [mode, selectedFile]);

  const isValidInlineText = mode === "text" && inlineText.trim().length > 0;

  const isValidFile = useMemo(() => {
    if (mode !== "file" || !selectedFile) return false;

    const ext = getFileExtension(selectedFile.name);
    const isAccepted = ACCEPTED_EXTENSIONS.includes(ext);
    const isWithinLimit = selectedFile.size <= MAX_FILE_SIZE_MB * 1024 * 1024;

    return isAccepted && isWithinLimit;
  }, [mode, selectedFile]);

  const canSubmit =
    !isSubmitting && ((mode === "file" && isValidFile) || isValidInlineText);

  function resetResultState() {
    setSummaryResult("");
    setOutputExtension("");
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
        `Unsupported file type: ${ext || "unknown"}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.`,
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      rejectFile(
        `File is too large. Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`,
      );
      return;
    }

    setError("");
    setSelectedFile(file);
    setOutputExtension(ext);
    setSummaryResult("");
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

  function switchToFileMode() {
    setMode("file");
    setInlineText("");
    setError("");
    resetResultState();
  }

  function switchToTextMode() {
    setMode("text");
    setSelectedFile(null);
    setError("");
    setOutputExtension(".txt");
    setSummaryResult("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!canSubmit) return;

    setIsSubmitting(true);
    setError("");
    resetResultState();

    try {
      let payload;

      if (mode === "text") {
        payload = {
          inputType: "text",
          inputExtension: ".txt",
          outputExtension: ".txt",
          text: inlineText.trim(),
        };
      } else {
        const ext = getFileExtension(selectedFile.name);

        payload = {
          inputType: "file",
          inputExtension: ext,
          outputExtension: ext,
          filename: selectedFile.name,
          // Replace this with real upload handling / FormData when wiring your API.
        };
      }

      // Replace this mock section with your real API call.
      // Example:
      // const formData = new FormData();
      // if (mode === "file") {
      //   formData.append("file", selectedFile);
      //   formData.append("inputExtension", payload.inputExtension);
      //   formData.append("outputExtension", payload.outputExtension);
      // } else {
      //   formData.append("text", payload.text);
      //   formData.append("inputExtension", ".txt");
      //   formData.append("outputExtension", ".txt");
      // }
      // const response = await fetch("/api/summarize", { method: "POST", body: formData });

      await new Promise((resolve) => setTimeout(resolve, 900));

      const mockSummary =
        mode === "text"
          ? `Summary generated from inline text.\n\nOutput extension: ${payload.outputExtension}\n\n${inlineText
              .trim()
              .slice(0, 240)}${inlineText.trim().length > 240 ? "..." : ""}`
          : `Summary generated from ${selectedFile.name}.\n\nInput extension: ${payload.inputExtension}\nOutput extension: ${payload.outputExtension}\n\nThe output format remains the same as the uploaded file format.`;

      setSummaryResult(mockSummary);
      setOutputExtension(payload.outputExtension);
    } catch (submitError) {
      setError("Something went wrong while generating the summary.");
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
            Back
          </button>

          <section className="mb-10">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-200 backdrop-blur">
              <Sparkles className="h-4 w-4" />
              Summarize content
            </div>

            <div className="mt-6 max-w-3xl">
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                Summarize documents or text with strict format rules
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-white/70 md:text-lg">
                Upload a PDF or Word document, or paste inline text. Unsupported
                files like PNG are automatically rejected, and the output
                extension always matches the input extension.
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
                <div className="mb-6 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={switchToFileMode}
                    className={`inline-flex items-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium transition ${
                      mode === "file"
                        ? "bg-white text-slate-900"
                        : "border border-white/10 bg-white/10 text-white/75 hover:bg-white/15 hover:text-white"
                    }`}
                  >
                    <FileType className="h-4 w-4" />
                    Upload file
                  </button>

                  <button
                    type="button"
                    onClick={switchToTextMode}
                    className={`inline-flex items-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium transition ${
                      mode === "text"
                        ? "bg-white text-slate-900"
                        : "border border-white/10 bg-white/10 text-white/75 hover:bg-white/15 hover:text-white"
                    }`}
                  >
                    <Type className="h-4 w-4" />
                    Inline text
                  </button>
                </div>

                {mode === "file" ? (
                  <div className="space-y-4">
                    <div
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      className="rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center transition hover:border-white/25 hover:bg-white/10"
                    >
                      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                        <Upload className="h-7 w-7 text-cyan-300" />
                      </div>

                      <h2 className="text-lg font-semibold text-white">
                        Upload a supported document
                      </h2>
                      <p className="mt-2 text-sm leading-6 text-white/65">
                        Allowed:{" "}
                        <span className="font-medium text-white">.pdf</span> and{" "}
                        <span className="font-medium text-white">.docx</span>.
                        Rejected automatically: .png, .jpg, and all unsupported
                        formats.
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
                        Choose file
                      </button>
                    </div>

                    {selectedFile && isValidFile && (
                      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                        <div className="flex items-start gap-3">
                          <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
                          <div>
                            <p className="font-medium text-emerald-100">
                              File accepted
                            </p>
                            <p className="mt-1 text-sm text-emerald-100/80">
                              {selectedFile.name} •{" "}
                              {formatBytes(selectedFile.size)}
                            </p>
                            <p className="mt-1 text-sm text-emerald-100/80">
                              Output extension will be{" "}
                              <span className="font-semibold">
                                {getFileExtension(selectedFile.name)}
                              </span>
                              .
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <label className="block">
                      <span className="mb-3 block text-sm font-medium text-white/80">
                        Paste text to summarize
                      </span>
                      <textarea
                        value={inlineText}
                        onChange={(e) => {
                          setInlineText(e.target.value);
                          setError("");
                          resetResultState();
                          setOutputExtension(".txt");
                        }}
                        rows={14}
                        placeholder="Paste or type your text here..."
                        className="w-full rounded-3xl border border-white/10 bg-white/5 px-5 py-4 text-sm leading-7 text-white outline-none placeholder:text-white/35 transition focus:border-cyan-300/40 focus:bg-white/10"
                      />
                    </label>

                    <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-4">
                      <p className="text-sm text-cyan-100">
                        Inline text is treated as{" "}
                        <span className="font-semibold">.txt</span>, so the
                        output extension will also be{" "}
                        <span className="font-semibold">.txt</span>.
                      </p>
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
                    {isSubmitting ? "Generating summary..." : "Summarize"}
                  </button>

                  <div className="text-sm text-white/55">
                    Output format:{" "}
                    <span className="font-medium text-white/85">
                      {derivedInputExtension || "—"}
                    </span>
                  </div>
                </div>
              </div>
            </form>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/10">
                    <FileText className="h-5 w-5 text-cyan-300" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      Format policy
                    </h2>
                    <p className="text-sm text-white/55">
                      Strict input and output matching
                    </p>
                  </div>
                </div>

                <div className="space-y-3 text-sm leading-6 text-white/70">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p>
                      <span className="font-semibold text-white">
                        Allowed uploads:
                      </span>{" "}
                      .pdf, .docx
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p>
                      <span className="font-semibold text-white">
                        Inline input:
                      </span>{" "}
                      treated as .txt
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p>
                      <span className="font-semibold text-white">
                        Rejected automatically:
                      </span>{" "}
                      .png and all unsupported file types
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p>
                      <span className="font-semibold text-white">
                        Output rule:
                      </span>{" "}
                      output extension must always equal input extension
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <h2 className="text-lg font-semibold text-white">
                  Summary output
                </h2>
                <p className="mt-1 text-sm text-white/55">Preview area</p>

                <div className="mt-4 rounded-2xl border border-white/10 bg-[#081120] p-4">
                  {summaryResult ? (
                    <>
                      <div className="mb-3 text-xs uppercase tracking-[0.18em] text-cyan-300/90">
                        Output extension: {outputExtension}
                      </div>
                      <pre className="whitespace-pre-wrap break-words text-sm leading-7 text-white/80">
                        {summaryResult}
                      </pre>
                    </>
                  ) : (
                    <p className="text-sm leading-6 text-white/45">
                      Your generated summary will appear here. The output
                      extension will always mirror the original input extension.
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

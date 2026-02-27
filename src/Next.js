"use client";

import { useMemo, useState } from "react";

const FEATURES = [
  { value: "summarize", label: "Summarize" },
  { value: "grammar_correct", label: "Grammar correction" },
  { value: "translate", label: "Translate" },
  { value: "explain", label: "Explain" },
  { value: "generate_questions", label: "Generate questions" },
  { value: "generate_answers", label: "Generate answers" },
];

function countWords(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

function parseNumberedLinesToArray(text) {
  // Accepts either:
  // 1) A newline-separated numbered list
  // 2) A plain list of lines (we'll auto-number on send)
  const lines = (text || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length === 0) return [];

  // If already numbered "1. ..." keep as-is; else auto-number.
  const alreadyNumbered = lines.every((l, idx) => l.startsWith(`${idx + 1}.`));

  if (alreadyNumbered) return lines;

  return lines.map((l, idx) => `${idx + 1}. ${l}`);
}

export default function HomePage() {
  const apiBase =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

  const [feature, setFeature] = useState("summarize");
  const [text, setText] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [questionsText, setQuestionsText] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState(null);

  const wordCount = useMemo(() => countWords(text), [text]);

  const derivedQuestions = useMemo(
    () => parseNumberedLinesToArray(questionsText),
    [questionsText],
  );

  const requirements = useMemo(() => {
    const missing = [];

    if (!text.trim()) missing.push("Paste or type document text.");

    if (feature === "translate" && !targetLanguage.trim()) {
      missing.push("Target language is required for translation.");
    }

    if (feature === "generate_questions" && wordCount <= 0) {
      missing.push("Valid text is required to compute word count.");
    }

    if (feature === "generate_answers" && derivedQuestions.length === 0) {
      missing.push("Questions are required for answer generation.");
    }

    // Your backend has MAX_WORD_COUNT=1000; the frontend warns early.
    // (Backend also enforces it.)
    if (wordCount > 1000) {
      missing.push("Text exceeds 1000-word limit (v1). Reduce content.");
    }

    return missing;
  }, [feature, text, targetLanguage, derivedQuestions.length, wordCount]);

  async function runProcess() {
    setError(null);
    setResult("");

    if (requirements.length > 0) {
      setError({
        title: "Fix required fields",
        message: requirements.join(" "),
      });
      return;
    }

    const payload = {
      text,
      feature,
    };

    if (feature === "generate_questions") {
      payload.word_count = wordCount;
    }

    if (feature === "translate") {
      payload.target_language = targetLanguage.trim();
    }

    if (feature === "generate_answers") {
      payload.questions = derivedQuestions;
      // Optional: include word_count too if you want future scaling logic;
      // your backend doesn't require it for answers currently. :contentReference[oaicite:6]{index=6}
    }

    setIsLoading(true);
    try {
      const res = await fetch(`${apiBase}/api/v1/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // If you later enable Auth0 in the frontend, add:
          // Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(payload),
      });

      const retryAfter = res.headers.get("Retry-After");

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        // Your backend wraps HTTPException as { detail: ... }
        const detail = data?.detail;

        // Normalize common shapes
        const message =
          typeof detail === "string"
            ? detail
            : detail?.message ||
              detail?.error ||
              data?.error ||
              "Request failed.";

        const extra = retryAfter
          ? ` Retry after ${retryAfter} seconds.`
          : detail?.retry_after_seconds
            ? ` Retry after ${detail.retry_after_seconds} seconds.`
            : "";

        setError({
          title: `Error (${res.status})`,
          message: `${message}${extra}`,
          raw: data,
        });
        return;
      }

      setResult(data?.result || "");
    } catch (e) {
      setError({
        title: "Network error",
        message: e?.message || "Failed to reach API.",
      });
    } finally {
      setIsLoading(false);
    }
  }

  function copyResult() {
    if (!result) return;
    navigator.clipboard.writeText(result).catch(() => {});
  }

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: 20,
        fontFamily: "system-ui",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 16,
          alignItems: "center",
        }}
      >
        <div>
          <h1 style={{ margin: 0 }}>jupitAIx — Document Processor (MVP)</h1>
          <p style={{ marginTop: 6, color: "#555" }}>
            Text-first frontend for your <code>/api/v1/process</code> endpoint.
          </p>
        </div>
        <div style={{ fontSize: 13, color: "#555" }}>
          API: <code>{apiBase}</code>
        </div>
      </header>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          marginTop: 16,
        }}
      >
        {/* Left: Inputs */}
        <div
          style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 14 }}
        >
          <div
            style={{
              display: "flex",
              gap: 10,
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <label style={{ fontWeight: 600 }}>Feature</label>
            <select
              value={feature}
              onChange={(e) => {
                setFeature(e.target.value);
                setError(null);
                setResult("");
              }}
              style={{ padding: 8, borderRadius: 8, border: "1px solid #ddd" }}
            >
              {FEATURES.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginTop: 12 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
              }}
            >
              <label style={{ fontWeight: 600 }}>Document text</label>
              <span
                style={{
                  fontSize: 12,
                  color: wordCount > 1000 ? "#b00020" : "#555",
                }}
              >
                Word count: {wordCount} / 1000
              </span>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your document text here..."
              rows={14}
              style={{
                width: "100%",
                marginTop: 6,
                padding: 10,
                borderRadius: 10,
                border: "1px solid #ddd",
                resize: "vertical",
              }}
            />
          </div>

          {feature === "translate" && (
            <div style={{ marginTop: 12 }}>
              <label style={{ fontWeight: 600 }}>Target language</label>
              <input
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                placeholder="e.g., French, Yoruba, Swahili"
                style={{
                  width: "100%",
                  marginTop: 6,
                  padding: 10,
                  borderRadius: 10,
                  border: "1px solid #ddd",
                }}
              />
              <p style={{ fontSize: 12, color: "#555", marginTop: 6 }}>
                Required for translation. :contentReference[oaicite:8]
                {(index = 8)}
              </p>
            </div>
          )}

          {feature === "generate_answers" && (
            <div style={{ marginTop: 12 }}>
              <label style={{ fontWeight: 600 }}>
                Questions (numbered list)
              </label>
              <textarea
                value={questionsText}
                onChange={(e) => setQuestionsText(e.target.value)}
                placeholder={`1. Question one...\n2. Question two...\n3. ...`}
                rows={8}
                style={{
                  width: "100%",
                  marginTop: 6,
                  padding: 10,
                  borderRadius: 10,
                  border: "1px solid #ddd",
                  resize: "vertical",
                }}
              />
              <p style={{ fontSize: 12, color: "#555", marginTop: 6 }}>
                Required for answer generation. The UI will auto-number if you
                paste plain lines.
              </p>
            </div>
          )}

          {error && (
            <div
              style={{
                marginTop: 12,
                padding: 10,
                borderRadius: 10,
                border: "1px solid #f3c2c2",
                background: "#fff6f6",
              }}
            >
              <div style={{ fontWeight: 700, color: "#b00020" }}>
                {error.title}
              </div>
              <div style={{ marginTop: 6, color: "#6b000f" }}>
                {error.message}
              </div>
            </div>
          )}

          <button
            onClick={runProcess}
            disabled={isLoading}
            style={{
              marginTop: 12,
              width: "100%",
              padding: 12,
              borderRadius: 10,
              border: "1px solid #111",
              background: isLoading ? "#eee" : "#111",
              color: isLoading ? "#111" : "#fff",
              cursor: isLoading ? "not-allowed" : "pointer",
              fontWeight: 700,
            }}
          >
            {isLoading ? "Processing..." : "Run"}
          </button>

          <div style={{ marginTop: 10, fontSize: 12, color: "#555" }}>
            Notes:
            <ul style={{ marginTop: 6 }}>
              <li>
                Question generation sends <code>word_count</code> computed
                client-side.{" "}
              </li>
              <li>
                Rate limiting errors show <code>Retry-After</code> when
                present.{" "}
              </li>
            </ul>
          </div>
        </div>

        {/* Right: Output */}
        <div
          style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 14 }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <h2 style={{ margin: 0, fontSize: 18 }}>Output</h2>
            <button
              onClick={copyResult}
              disabled={!result}
              style={{
                padding: "8px 10px",
                borderRadius: 10,
                border: "1px solid #ddd",
                background: result ? "#fff" : "#f6f6f6",
                cursor: result ? "pointer" : "not-allowed",
              }}
            >
              Copy
            </button>
          </div>

          <textarea
            value={result}
            readOnly
            placeholder="Your result will appear here..."
            rows={24}
            style={{
              width: "100%",
              marginTop: 10,
              padding: 10,
              borderRadius: 10,
              border: "1px solid #ddd",
              resize: "vertical",
              fontFamily:
                "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fontSize: 13,
            }}
          />

          <div style={{ marginTop: 10, fontSize: 12, color: "#555" }}>
            If you later add Auth0 login, send the bearer token in the
            Authorization header; your backend already supports optional auth.
          </div>
        </div>
      </section>

      <footer style={{ marginTop: 18, color: "#666", fontSize: 12 }}>
        MVP UI (text-first). Next step: add file upload once your backend wires
        extraction/OCR into an upload endpoint.
      </footer>
    </main>
  );
}

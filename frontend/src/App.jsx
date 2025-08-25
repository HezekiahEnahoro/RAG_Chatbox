import React, { useRef, useState } from "react";
import "./App.css"
const API = import.meta.env.VITE_API || "http://localhost:8001";

export default function Chat() {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  async function onUpload() {
    const files = fileRef.current?.files;
    if (!files || !files.length) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    setBusy(true);
    try {
      const r = await fetch(`${API}/ingest`, { method: "POST", body: fd });
    if (r.status === 429){
      const data = await r.json()
      setError(data?.message || "You’ve hit the upload rate limit.");
      return;  
      }
    if (!r.ok) {
      const text = await r.text(); // might be empty/HTML
      setError(`Upload failed (${r.status}). ${text?.slice(0, 200) || ""}`);
      return;
    }
      const ct = r.headers.get("content-type") || "";
      const j = ct.includes("application/json") ? await r.json() : null;
      if (!j) {
        setError("Unexpected server response.");
        return;
      }

      pushMsg({
        role: "system",
        content: `Ingested ${j.chunks_added} chunks from: ${j.files.join(
          ", "
        )}`,
      });
    } finally {
      setBusy(false);
    }
  }

  function pushMsg(m) {
    setMsgs((s) => [...s, m]);
  }

  async function onAsk(e) {
    e.preventDefault();
    const text = q.trim();
    if (!text) return;
    pushMsg({ role: "user", content: text });
    setQ("");
    setBusy(true);
    try {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      const j = await r.json();
      const answer = j.ok ? j.answer : j.error || "Error";
    
      pushMsg({ role: "assistant", content: `${answer}\n\n` });
    } catch (err) {
      pushMsg({ role: "assistant", content: `Error: ${String(err)}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-bold">RAG Chatbot</h1>

      <div className="rounded-xl border bg-white p-4">
        <div className="flex gap-2 items-center">
          <input
            ref={fileRef}
            type="file"
            multiple
            className="block w-full text-sm"
            accept=".pdf,.txt,.md"
          />

          <button
            onClick={onUpload}
            disabled={busy}
            className="px-3 py-2 rounded bg-indigo-600 text-white disabled:opacity-50">
            Upload
          </button>
          {error && (
            <div className="rounded-md border border-red-300 bg-red-50 p-2 text-sm">
              {error} <strong>Try again in an hour</strong>
            </div>
          )}
        </div>
      </div>
      <div className="">
        {busy ? (
          <div
            role="status"
            aria-live="polite"
            className="grid place-items-center gap-2 rounded-xl border bg-white p-4">
            {/* thinking: three dots + progress sweep */}
            <div className="flex items-end gap-1" aria-hidden="true">
              <span
                className="h-2 w-2 rounded-full bg-gray-900 opacity-25"
                style={{ animation: "dots 1.2s ease-in-out infinite" }}
              />
              <span
                className="h-2 w-2 rounded-full bg-gray-900 opacity-25"
                style={{ animation: "dots 1.2s .15s ease-in-out infinite" }}
              />
              <span
                className="h-2 w-2 rounded-full bg-gray-900 opacity-25"
                style={{ animation: "dots 1.2s .3s ease-in-out infinite" }}
              />
            </div>
            <div
              className="h-1.5 w-40 overflow-hidden rounded-full bg-gray-200"
              aria-hidden="true">
              <span
                className="block h-full w-1/3 rounded-full bg-gray-900"
                style={{ animation: "slide 1.6s ease-in-out infinite" }}
              />
            </div>
            <div className="text-xs text-gray-600">thinking…</div>
          </div>
        ) : (
          <div
            role="status"
            aria-live="polite"
            className="grid place-items-center gap-3 rounded-xl border bg-white p-6">
            {/* idle: soft floating blob + shimmer line */}
            <div
              className="h-7 w-7 rounded-full"
              style={{
                background:
                  "radial-gradient(circle at 30% 30%, #7aa7ff, #6ef3d6)",
                filter: "blur(.2px)",
                animation: "idle-float 3.2s ease-in-out infinite",
              }}
            />
            <div
              className="h-2 w-40 rounded-full"
              style={{
                background:
                  "linear-gradient(90deg, rgba(0,0,0,.08) 20%, rgba(0,0,0,.18) 40%, rgba(0,0,0,.08) 60%)",
                backgroundSize: "200% 100%",
                animation: "shimmer 2.2s linear infinite",
              }}
            />
            <div className="text-xs text-gray-600">
              welcome… ask me anything
            </div>
          </div>
        )}
      </div>

      <div className="rounded-xl border bg-white p-4 h-[40vh] overflow-y-auto">
        {msgs.length === 0 && (
          <div className="text-sm text-gray-500">
            Upload docs and ask a question (e.g., “What are the key dates?”).
          </div>
        )}
        {busy && msgs.length === 0 && (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="h-4 w-2/3 rounded bg-gray-200"
                style={{
                  animation: "shimmer 2.2s linear infinite",
                  backgroundImage:
                    "linear-gradient(90deg, rgba(0,0,0,.06) 20%, rgba(0,0,0,.14) 40%, rgba(0,0,0,.06) 60%)",
                  backgroundSize: "200% 100%",
                }}
              />
            ))}
          </div>
        )}

        {msgs.map((m, i) => (
          <div
            key={i}
            className={`mb-3 ${m.role === "user" ? "text-right" : ""}`}>
            <div
              className={`inline-block px-3 py-2 rounded-lg ${
                m.role === "user" ? "bg-indigo-600 text-white" : "bg-gray-100"
              }`}>
              {m.content}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={onAsk} className="flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask a question about your documents…"
          className="w-full rounded-lg border px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button
          type="submit"
          className="px-4 py-2 rounded bg-indigo-600 text-white">
          Send
        </button>
      </form>
    </div>
  );
}

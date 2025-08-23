import React, { useRef, useState } from "react";
import "./App.css"
const API = import.meta.env.VITE_API || "http://localhost:8001";

export default function Chat() {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);

  async function onUpload() {
    const files = fileRef.current?.files;
    if (!files || !files.length) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    setBusy(true);
    try {
      const r = await fetch(`${API}/ingest`, { method: "POST", body: fd });
      const j = await r.json();
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
            {busy ? "Working…" : "Ingest"}
          </button>
        </div>
      </div>

      <div className="rounded-xl border bg-white p-4 h-[60vh] overflow-y-auto">
        {msgs.length === 0 && (
          <div className="text-sm text-gray-500">
            Upload docs and ask a question (e.g., “What are the key dates?”).
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
        <button className="px-4 py-2 rounded bg-indigo-600 text-white">
          Send
        </button>
      </form>
    </div>
  );
}

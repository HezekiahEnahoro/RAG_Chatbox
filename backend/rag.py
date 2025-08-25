# rag.py (memory-friendly)
import os, json, uuid
from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
from openai import OpenAI
# Optional TF-IDF fallback (kept light)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- ENV toggles (set these on Render) ----------
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND")
BACKEND = os.getenv("EMBEDDINGS_BACKEND", "openai").lower()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")  # e.g. /app/cache/hf
STORE_DIR = os.getenv("STORE_DIR", "store")

# Strongly limit hidden thread pools (saves RAM/CPU)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------------------------------

class VectorStore:
    def __init__(self, dim: int, store_dir: str, collection: str = "default"):
        self.collection = collection
        self.store_dir = os.path.join(store_dir, collection)
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "index.faiss")
        self.meta_path  = os.path.join(self.store_dir, "meta.jsonl")
        self.dim = dim
        self.index = None
        self.meta_count = 0  # track rows without loading entire meta file in RAM
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        # count meta lines without holding them all
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta_count = sum(1 for _ in f)
        else:
            self.meta_count = 0

    def _append_meta(self, metadatas: List[Dict]):
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for m in metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self.meta_count += len(metadatas)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict], save_every: int = 2048):
        # normalize for cosine via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        emb_norm = (embeddings / norms).astype("float32", copy=False)
        self.index.add(emb_norm)
        self._append_meta(metadatas)
        # save incrementally to keep memory stable
        if self.index.ntotal % save_every < embeddings.shape[0]:
            faiss.write_index(self.index, self.index_path)

    def _read_meta_at(self, idxs: List[int]) -> List[Dict]:
        # random access into JSONL: stream lines; fine for small k (<= 50)
        out, want = [], set(idxs)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in want:
                    out.append(json.loads(line))
                    if len(out) == len(want): break
        return out

    def search(self, query_emb: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(q.astype("float32"), k)
        idxs = [i for i in I[0] if i >= 0]
        metas = self._read_meta_at(idxs) if idxs else []
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            out.append((float(score), metas[idxs.index(idx)]))
        return out


class RAGPipeline:
    def __init__(self):
        # Don’t load SBERT here; do it lazily
        self._sbert = None
        self._dim: Optional[int] = None
        self._stores: Dict[str, VectorStore] = {}
        self._tfidf: Dict[str, Dict] = {}

        # If OpenAI, set vector dim lazily to known sizes only when needed.
        # For SBERT we’ll read from the model once it loads.

    # ---------- SBERT lazy loader ----------
    def _get_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=TRANSFORMERS_CACHE)
            self._dim = int(self._sbert.get_sentence_embedding_dimension())
        return self._sbert

    # ---------- Vector store per collection ----------
    def _get_store(self, collection: str) -> VectorStore:
            # # Set a default dim if still unknown (OpenAI small=1536). Adjust if you use -large.
            # if BACKEND == "openai":
            #     self._dim = 1536
            # else:
            #     # For SBERT, force model load to learn dim
            #     self._dim = int(self._get_sbert().get_sentence_embedding_dimension())
        vs = self._stores.get(collection)
        if not vs:
            if self._dim is None:
                 raise RuntimeError("Vector dimension unknown; call embed_texts once before creating a store.")
            vs = VectorStore(dim=self._dim, store_dir=STORE_DIR, collection=collection)
            self._stores[collection] = vs
        return vs

    # ---------- Chunking ----------
    def chunk_text(self, text: str, max_tokens: int = 400, overlap: int = 80) -> List[str]:
        words = text.split()
        chunks, cur, cur_len = [], [], 0
        for w in words:
            cur.append(w); cur_len += len(w) + 1
            if cur_len >= max_tokens:
                s = " ".join(cur)
                chunks.append(s)
                back = s[-overlap:] if overlap else ""
                cur = [back] if back else []
                cur_len = len(back)
        if cur:
            chunks.append(" ".join(cur))
        return [c.strip() for c in chunks if len(c.strip()) > 20]

    # ---------- Embeddings ----------
    def _embed_openai(self, texts):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI()
        out = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        embs = np.array([e.embedding for e in out.data], dtype="float32")
        return embs

    def _embed_sbert(self, texts: List[str]) -> np.ndarray:
        model = self._get_sbert()
        # small batch, no progress bar → lower RAM
        embs = model.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        return embs.astype("float32", copy=False)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # Try OpenAI first unless explicitly forced to SBERT
        if BACKEND == "sbert":
            return self._embed_sbert(texts)
        try:
            return self._embed_openai(texts)
        except Exception as e:
            msg = str(e).lower()
            # Fallback on quota/no-key/429 etc.
            if ("insufficient_quota" in msg or "quota" in msg or "429" in msg
                or "openai_api_key" in msg or "not set" in msg):
                # Switch to SBERT transparently
                return self._embed_sbert(texts)
            raise

    # ---------- Ingest ----------
    def ingest(self, docs, collection: str = "default", batch: int = 256) -> int:
        chunks, metas = [], []
        for doc_id, text in docs:
            for i, chunk in enumerate(self.chunk_text(text)):
                chunks.append(chunk)
                metas.append({"id": str(uuid.uuid4()), "doc_id": doc_id, "chunk_id": i, "text": chunk})

        if not chunks:
            return 0

        total = 0
        start = 0
        while start < len(chunks):
            end = min(start + batch, len(chunks))
            batch_chunks = chunks[start:end]
            batch_meta   = metas[start:end]

            embs = self.embed_texts(batch_chunks)  # may be OpenAI OR fallback SBERT
            if self._dim is None:
                self._dim = int(embs.shape[1])     # decide dim now, based on real backend
            store = self._get_store(collection)

            # cosine via inner product: normalize before adding (VectorStore also normalizes)
            store.add(embs, batch_meta)
            total += len(batch_chunks)
            start = end

        faiss.write_index(self._stores[collection].index, self._stores[collection].index_path)
        return total


    # ---------- Retrieve ----------
    def retrieve(self, query: str, collection: str = "default", k: int = 5):
        if BACKEND == "tfidf":
            b = self._tfidf.get(collection)
            if not b or b["matrix"] is None or not b["texts"]: return []
            qv = sk_normalize(b["vectorizer"].transform([query]))
            sims = (b["matrix"] @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [(float(sims[i]), b["meta"][i]) for i in idx]
        # vector path
        q = self.embed_texts([query])
        store = self._get_store(collection)
        return store.search(q, k=k)

    # ---------- Answer ----------
    def answer(self, query: str, collection: str = "default", use_openai: bool = True) -> Dict:
        hits = self.retrieve(query, collection=collection, k=5)
        seen, uniq = set(), []
        for s, m in hits:
            key = (m["doc_id"], m["chunk_id"])
            if key not in seen:
                seen.add(key); uniq.append((s, m))
        context = "\n\n".join([m["text"] for _, m in uniq])
        sources = [{"doc_id": m["doc_id"], "chunk_id": m["chunk_id"], "score": s} for s, m in uniq]

        if not use_openai or os.getenv("OPENAI_API_KEY") is None:
            return {"ok": True, "answer": "Relevant excerpts:\n" + context, "sources": sources}

        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "You are a helpful assistant. Answer the question using ONLY the context below. "
                "Do NOT include bracketed citations like [1] or [2].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely."
            )
            resp = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=400,
            )
            return {"ok": True, "answer": resp.choices[0].message.content, "sources": sources}
        except Exception as e:
            # Graceful fallback: return stitched context if OpenAI answer fails (e.g., quota)
            return {"ok": True, "answer": "Relevant excerpts:\n" + context, "sources": sources, "note": str(e)}

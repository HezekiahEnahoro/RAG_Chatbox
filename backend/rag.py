# rag.py (replace RAGPipeline with this version)
import os, json, uuid
from typing import List, Tuple, Dict
import numpy as np
import faiss

# NEW imports for fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STORE_DIR = os.getenv("STORE_DIR", "store")

class VectorStore:
    def __init__(self, dim: int, store_dir: str, collection: str = "default"):
        self.base_dir = store_dir
        self.collection = collection
        self.store_dir = os.path.join(store_dir, collection)
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "index.faiss")
        self.meta_path  = os.path.join(self.store_dir, "meta.jsonl")
        self.dim = dim
        self.index = None
        self.meta: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = [json.loads(line) for line in f]
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        emb_norm = embeddings / norms
        self.index.add(emb_norm.astype("float32"))
        self.meta.extend(metadatas)
        self._save()

    def search(self, query_emb: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12
        q = (query_emb / norms).astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((float(score), self.meta[idx]))
        return out


class RAGPipeline:
    def __init__(self):
        self.use_tfidf = False
        # TF-IDF per collection
        self._tfidf_by_coll = {}  # coll -> dict(vectorizer, matrix, texts, meta)
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(EMBED_MODEL_NAME)
            self.dim = self.sbert.get_sentence_embedding_dimension()
            self._stores = {}  # coll -> VectorStore
        except Exception as e:
            print(f"[RAG] SBERT unavailable ({e}). Falling back to TF-IDF.")
            self.use_tfidf = True

    def _get_store(self, collection: str):
        if self.use_tfidf:
            d = self._tfidf_by_coll.setdefault(collection, {
                "vectorizer": None, "matrix": None, "texts": [], "meta": []
            })
            if d["vectorizer"] is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
                d["vectorizer"] = TfidfVectorizer(
                    max_features=4096, ngram_range=(1,2), lowercase=True, stop_words="english"
                )
                d["normalize"] = normalize
            return d
        else:
            if collection not in self._stores:
                self._stores[collection] = VectorStore(dim=self.dim, store_dir=STORE_DIR, collection=collection)
            return self._stores[collection]


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

    # === SBERT path ===
    def embed_texts_sbert(self, texts: List[str]) -> np.ndarray:
        embs = self.sbert.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return embs

    # === TF-IDF path (fallback) ===
    def _refit_tfidf(self):
        if not self.vectorizer:
            raise RuntimeError("TF-IDF vectorizer not initialized")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        self.tfidf_matrix = normalize(self.tfidf_matrix)  # cosine

    def ingest(self, docs, collection: str = "default"):
        chunks, metas = [], []
        for doc_id, text in docs:
            for i, chunk in enumerate(self.chunk_text(text)):
                chunks.append(chunk)
                metas.append({"doc_id": doc_id, "chunk_id": i, "text": chunk})

        if not chunks:
            return 0

        if not self.use_tfidf:
            embs = self.embed_texts_sbert(chunks)
            # add stable ids if you like
            metas = [{"id": str(uuid.uuid4()), **m} for m in metas]
            store: VectorStore = self._get_store(collection)
            store.add(embs, metas)
        else:
            bucket = self._get_store(collection)
            bucket["texts"].extend(chunks)
            bucket["meta"].extend(metas)
            V = bucket["vectorizer"]
            bucket["matrix"] = bucket["normalize"](V.fit_transform(bucket["texts"]))
        return len(chunks)

    def retrieve(self, query: str, collection: str = "default", k: int = 5):
        if not self.use_tfidf:
            q = self.embed_texts_sbert([query])
            store: VectorStore = self._get_store(collection)
            return store.search(q, k=k)
        else:
            b = self._get_store(collection)
            if b["matrix"] is None or not b["texts"]:
                return []
            qv = b["normalize"](b["vectorizer"].transform([query]))
            sims = (b["matrix"] @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            results = [(float(sims[i]), b["meta"][i]) for i in idx]
            return results

    
    def answer(self, query: str, collection: str = "default", use_openai: bool = True) -> Dict:
            hits = self.retrieve(query, collection=collection, k=5)
            # dedupe sources
            seen, uniq = set(), []
            for s, m in hits:
                key = (m["doc_id"], m["chunk_id"])
                if key not in seen:
                    seen.add(key); uniq.append((s, m))
            context = "\n\n".join([m["text"] for _, m in uniq])
            sources = [{"doc_id": m["doc_id"], "chunk_id": m["chunk_id"], "score": s} for s, m in uniq]

            if not use_openai or os.getenv("OPENAI_API_KEY") is None:
                return {"ok": True, "answer": "Relevant excerpts:\n" + context, "sources": sources}

            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "You are a helpful assistant. Answer the question using ONLY the context below. "
                "Do NOT include bracketed citations or numbers like [1], [2] in your answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer concisely in plain sentences."
                )

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.2, max_tokens=400,
            )
            return {"ok": True, "answer": resp.choices[0].message.content, "sources": sources}
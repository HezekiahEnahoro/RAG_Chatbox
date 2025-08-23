# RAG Chatbox (FastAPI)

Retrieval-Augmented Generation chatbox that ingests your documents, stores embeddings in a vector DB, and answers questions grounded in your data.

## Features
- Upload PDFs/Docs → chunk → embed → vector search
- FastAPI backend with streaming responses
- Pluggable vector store (FAISS)
- Clean API routes: /ingest, /ask

## Quickstart
1) Clone & create env:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
2) Configure:
   cp .env.example .env  # fill in keys
3) Run:
   uvicorn app.main:app --reload

## Config
See `.env.example` for required variables.

## Storage
- Local dev: ./data (ignored by git)
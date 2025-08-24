import os, io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from markdown import markdown
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from rag import RAGPipeline
import uvicorn


app = FastAPI(title="RAG Chatbot API")
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX")  # e.g. r"https://.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_origin_regex=origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
    allow_credentials=os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true",
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx"}
MAX_QUERY_LENGTH = 500

os.makedirs("data", exist_ok=True)
rag = RAGPipeline()

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB")
    
    if file.filename:
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type. Allowed: {list(ALLOWED_EXTENSIONS)}")


class AskBody(BaseModel):
    query: str
    collection: Optional[str] = "default"
# class AskBody(BaseModel):
#     query: str
#     collection: str | None = None

    
@app.get("/")
@limiter.limit("60/minute")  # Liberal limit for health checks
async def health(request: Request):
    """Health check endpoint"""
    try:
        if hasattr(rag, 'vs') and rag.vs and hasattr(rag.vs, 'index'):
            total_docs = rag.vs.index.ntotal
        else:
            total_docs = 0
    except:
        total_docs = 0
    
    return {
        "ok": True, 
        "status": "running",
        "total_documents": total_docs,
        "message": "RAG Chatbot API is healthy",
        "rate_limits": {
            "uploads": "5 per hour",
            "queries": "20 per hour", 
            "health_checks": "60 per minute"
        }
    }

@app.post("/ingest")
@limiter.limit("5/hour")
async def ingest(request: Request, files: list[UploadFile] = File(...), collection: str = Query("default")):
    """Upload and ingest documents"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    if len(files) > 5:
        raise HTTPException(400, "Too many files. Maximum 5 files per request")
    
    docs = []
    saved_files = []
    
    try:
        for f in files:
            validate_file(f)
            content = await f.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"File {f.filename} too large")
            
            text = extract_text(f.filename, content)
            if text.strip():
                with open(os.path.join("data", f.filename), "wb") as w:
                    w.write(content)
                saved_files.append(f.filename)
                docs.append((f.filename, text))
            else:
                raise HTTPException(400, f"No text could be extracted from {f.filename}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing files: {str(e)}")
    
    if not docs:
        raise HTTPException(400, "No valid documents to ingest")
    
    try:
        n = rag.ingest(docs, collection=collection)
        return {"ok": True, "files": [f.filename for f in files], "chunks_added": n, "collection": collection}
    except Exception as e:
        raise HTTPException(500, f"Error ingesting documents: {str(e)}")
    

@app.post("/ask")
@limiter.limit("4/hour")
async def ask(request: Request, body: AskBody):
    """Ask questions about ingested documents"""
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    if len(body.query) > MAX_QUERY_LENGTH:
        raise HTTPException(400, f"Query too long. Maximum {MAX_QUERY_LENGTH} characters")
    
    collection = (body.collection or "default").strip()
    if not collection.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "Invalid collection name")
    
    try:
        # Check if collection exists and has data
        has_data = False
        
        if hasattr(rag, '_stores') and collection in rag._stores:
            store = rag._stores[collection]
            has_data = hasattr(store, 'index') and store.index.ntotal > 0
        elif hasattr(rag, '_tfidf_by_coll') and collection in rag._tfidf_by_coll:
            tfidf_data = rag._tfidf_by_coll[collection]
            has_data = bool(tfidf_data.get('texts'))
        
        if not has_data:
            return {
                "ok": False,
                "error": f"No documents found in collection '{collection}'. Please upload documents first.",
                "collection": collection
            }
        
        result = rag.answer(body.query, collection=collection, use_openai=True)
        result.update({
            "query": body.query,
            "collection": collection
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error processing query: {str(e)}")

def extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n\n".join(pages)
        elif name.endswith(".md"):
            # keep markdown plain (optionally convert to text)
            html = markdown(content.decode("utf-8", errors="ignore"))
            # naive strip tags (optional): remove '<...>'
            import re
            return re.sub(r"<[^>]+>", " ", html)
        else:
            # assume text
            return content.decode("utf-8", errors="ignore")
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from {filename}: {str(e)}")    

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False
    )
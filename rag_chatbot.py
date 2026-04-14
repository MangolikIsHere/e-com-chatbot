# =========================
# 1. FastAPI Setup
# =========================
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json
import shutil
from functools import lru_cache
from dotenv import load_dotenv
from pathlib import Path

app = FastAPI()

# Allow local frontend apps (e.g., Next.js dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DEBUG_ENDPOINTS_ENABLED = os.getenv("ENABLE_DEBUG_ENDPOINTS", "true").lower() in {
    "1",
    "true",
    "yes",
}

# =========================
# 2. Load ENV variables
# =========================
load_dotenv(BASE_DIR / ".env")

# =========================
# 3. Gemini Setup (NEW SDK)
# =========================
@lru_cache(maxsize=1)
def get_gemini_client():
    api_key_value = os.getenv("GOOGLE_API_KEY")
    if not api_key_value:
        raise RuntimeError("GOOGLE_API_KEY not found in .env file")

    from google import genai

    return genai.Client(api_key=api_key_value)


def ask_gemini(prompt):
    response = get_gemini_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.3,
            "max_output_tokens": 500
        }
    )
    return response.text or "No response"

# =========================
# 4. RAG Setup (PDF)
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def _resolve_pdf_path() -> Path:
    pdf_candidates = [BASE_DIR / "ml_book.pdf", BASE_DIR / "ml-book.pdf"]
    pdf_path = next((candidate for candidate in pdf_candidates if candidate.exists()), None)
    if pdf_path is None:
        raise FileNotFoundError("❌ No PDF found (ml_book.pdf or ml-book.pdf)")
    return pdf_path


def _index_metadata(pdf_path: Path) -> dict:
    return {
        "pdf_name": pdf_path.name,
        "pdf_size": pdf_path.stat().st_size,
        "pdf_mtime": int(pdf_path.stat().st_mtime),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def _load_saved_metadata(vector_store_dir: Path) -> dict | None:
    metadata_file = vector_store_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_metadata(vector_store_dir: Path, metadata: dict) -> None:
    metadata_file = vector_store_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _build_faiss_index(docs, embeddings, vector_store_dir: Path, metadata: dict) -> FAISS:
    db = FAISS.from_documents(docs, embeddings)
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(vector_store_dir))
    _save_metadata(vector_store_dir, metadata)
    return db


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    pdf_path = _resolve_pdf_path()

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store_dir = BASE_DIR / "faiss_index"
    current_metadata = _index_metadata(pdf_path)
    saved_metadata = _load_saved_metadata(vector_store_dir)

    should_rebuild = not vector_store_dir.exists() or saved_metadata != current_metadata

    if not should_rebuild:
        try:
            db = FAISS.load_local(
                str(vector_store_dir),
                embeddings,
                allow_dangerous_deserialization=True
            )
            if db.index.ntotal == 0:
                should_rebuild = True
        except Exception:
            should_rebuild = True

    if should_rebuild:
        db = _build_faiss_index(docs, embeddings, vector_store_dir, current_metadata)

    return db


@lru_cache(maxsize=1)
def get_retriever():
    db = get_vector_store()

    return db.as_retriever(search_kwargs={"k": 4})

# =========================
# 5. Session-based Memory
# =========================
chat_sessions = {}

def get_history(session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

def format_history(history):
    formatted = ""
    for q, a in history[-5:]:
        formatted += f"User: {q}\nBot: {a}\n"
    return formatted

# =========================
# 6. Request Model
# =========================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str


class DebugRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=10)

# =========================
# 7. Health Check Route
# =========================
@app.get("/")
def home():
    return {"message": "✅ ML Tutor Chatbot API running"}


@app.post("/debug/retrieval")
def debug_retrieval(request: DebugRetrievalRequest):
    if not DEBUG_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Debug endpoint disabled")

    try:
        db = get_vector_store()
        results = db.similarity_search_with_score(request.query, k=request.k)

        chunks = []
        for index, (doc, score) in enumerate(results, start=1):
            text = (doc.page_content or "").strip().replace("\n", " ")
            if len(text) > 350:
                text = f"{text[:350]}..."

            chunks.append(
                {
                    "rank": index,
                    "score": float(score),
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "text_preview": text,
                }
            )

        return {
            "query": request.query,
            "k": request.k,
            "result_count": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval debug failed: {str(e)}")

# =========================
# 8. Chat Endpoint
# =========================
@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message
    session_id = request.session_id

    try:
        # Get session memory
        history = get_history(session_id)

        # Retrieve context
        retriever = get_retriever()
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        # Format history
        history_text = format_history(history)

        # Mode detection
        if "simple" in query.lower() or "beginner" in query.lower():
            mode = "Explain in very simple terms like a beginner."
        else:
            mode = "Give a detailed explanation."

        # Prompt
        prompt = f"""
You are an expert Machine Learning tutor.

Instruction:
{mode}

Conversation History:
{history_text}

Context:
{context}

Rules:
- Answer ONLY from the context
- If not found, say "I don't know from given material"
- Keep answers clear and structured

Question:
{query}

Answer:
"""

        # Generate response
        answer = ask_gemini(prompt)

        # Save to memory
        history.append((query, answer))

        return {
            "response": answer,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
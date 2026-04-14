# =========================
# 1. FastAPI Setup
# =========================
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from pathlib import Path

app = FastAPI()

# =========================
# 2. Load ENV variables
# =========================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# =========================
# 3. Gemini Setup (NEW SDK)
# =========================
from google import genai

client = genai.Client(api_key=api_key)

def ask_gemini(prompt):
    response = client.models.generate_content(
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

pdf_candidates = [Path("ml_book.pdf"), Path("ml-book.pdf")]
pdf_path = next((c for c in pdf_candidates if c.exists()), None)

if pdf_path is None:
    raise FileNotFoundError("❌ No PDF found (ml_book.pdf or ml-book.pdf)")

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

vector_store_dir = "faiss_index"

if os.path.exists(vector_store_dir):
    db = FAISS.load_local(
        vector_store_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_store_dir)

retriever = db.as_retriever(search_kwargs={"k": 4})

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

# =========================
# 7. Health Check Route
# =========================
@app.get("/")
def home():
    return {"message": "✅ ML Tutor Chatbot API running"}

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
        return {"error": str(e)}
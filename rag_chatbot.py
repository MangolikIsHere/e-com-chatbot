# =========================
# 1. FastAPI Setup
# =========================
from fastapi import FastAPI


app = FastAPI()

# =========================
# 2. Load ENV variables
# =========================
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# =========================
# 3. Gemini Setup
# =========================
import google.generativeai as genai

model = None
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

def ask_gemini(prompt):
    if model is None:
        raise RuntimeError("GOOGLE_API_KEY not found. Add it to your .env file before calling /chat.")
    response = model.generate_content(prompt)
    return response.text

# =========================
# 4. RAG Setup (PDF)
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

pdf_candidates = [Path("ml_book.pdf"), Path("ml-book.pdf")]
pdf_path = next((candidate for candidate in pdf_candidates if candidate.exists()), None)

if pdf_path is None:
    raise FileNotFoundError("No PDF found. Expected ml_book.pdf or ml-book.pdf in project root.")

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
# 5. Chat Memory (per session simple)
# =========================
chat_history = []

def format_history(history):
    formatted = ""
    for q, a in history[-5:]:
        formatted += f"User: {q}\nBot: {a}\n"
    return formatted

# =========================
# 6. Request Model
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# 7. API Endpoint
# =========================
@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message

    try:
        # Retrieve context
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        # History
        history_text = format_history(chat_history)

        # Mode
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

Question:
{query}

Answer:
"""

        answer = ask_gemini(prompt)

        # Save memory
        chat_history.append((query, answer))

        return {
            "response": answer
        }

    except Exception as e:
        return {
            "error": str(e)
        }
# =========================
# 1. Load ENV variables
# =========================
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# =========================
# 2. Gemini (Official SDK)
# =========================
import google.generativeai as genai

genai.configure(api_key=api_key)

def _resolve_model_name(preferred_model: str) -> str:
    """Choose a model that supports generateContent, preferring user choice."""
    try:
        available = [
            m.name
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
    except Exception:
        available = []

    preferred_full = preferred_model if preferred_model.startswith("models/") else f"models/{preferred_model}"
    if preferred_full in available:
        return preferred_full

    # Stable default if model listing is unavailable.
    if not available:
        return "gemini-1.5-flash"

    # Prefer a flash model for speed/cost if present.
    for name in available:
        if "flash" in name.lower():
            return name

    return available[0]


preferred_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
resolved_model = _resolve_model_name(preferred_model)
model = genai.GenerativeModel(resolved_model)
print(f"Using model: {resolved_model}")

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

# =========================
# 3. Imports (LangChain for RAG)
# =========================
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# 4. Load document
# =========================
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# =========================
# 5. Split text
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# =========================
# 6. Embeddings (LOCAL)
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# 7. Vector Store
# =========================
vector_store_dir = "faiss_index"

if os.path.exists(vector_store_dir):
    print(f"Loading saved vector store from '{vector_store_dir}'...")
    db = FAISS.load_local(
        vector_store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    print("No saved vector store found. Creating and saving a new one...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_store_dir)

# Always retrieve from the saved-on-disk version.
db = FAISS.load_local(
    vector_store_dir,
    embeddings,
    allow_dangerous_deserialization=True,
)

# =========================
# 8. Retriever
# =========================
retriever = db.as_retriever(search_kwargs={"k": 3})

# =========================
# 9. Chat Loop
# =========================
print("🤖 Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    try:
        # 🔍 Retrieve relevant docs
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        # 🧠 Create prompt
        prompt = f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

        # 🤖 Get Gemini response
        answer = ask_gemini(prompt)

        print("Bot:", answer, "\n")

    except Exception as e:
        print("❌ Error:", str(e))
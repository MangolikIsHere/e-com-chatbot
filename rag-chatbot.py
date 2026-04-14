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

model = genai.GenerativeModel("gemini-1.0-pro")

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

# =========================
# 3. Imports (RAG)
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
# 6. Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# 7. Vector Store (Save + Load)
# =========================
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
        # Retrieve context
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
You are a helpful AI assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{query}
"""

        # Generate answer
        answer = ask_gemini(prompt)

        print("Bot:", answer, "\n")

    except Exception as e:
        print("❌ Error:", str(e))
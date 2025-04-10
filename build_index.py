from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# === Config ===
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Step 1: Load documents ===
def load_documents_from_pdfs(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# === Step 2: Split documents into text chunks ===
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# === Step 3: Load embedding model ===
def load_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

# === Step 4: Create FAISS vector store ===
def build_vectorstore(text_chunks, embedding_model, db_path):
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    return db

# === Main execution ===
if __name__ == "__main__":
    print("[*] Loading documents...")
    docs = load_documents_from_pdfs(DATA_PATH)

    print(f"[*] Loaded {len(docs)} documents. Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"[*] Created {len(chunks)} chunks. Loading embedding model...")
    embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    print("[*] Building FAISS vectorstore...")
    db = build_vectorstore(chunks, embed_model, DB_FAISS_PATH)

    print(f"[✓] FAISS index saved to '{DB_FAISS_PATH}'")

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Config ===
DATA_PATH = "data/"
INDEX_SAVE_PATH = "rag_index.pkl"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# === Step 1: Load documents ===
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# === Step 2: Split documents into chunks ===
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# === Step 3: Build FAISS index ===
def build_faiss_index(docs, model_name):
    embedder = SentenceTransformer(model_name)
    texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(INDEX_SAVE_PATH, "wb") as f:
        pickle.dump((index, texts), f)

    print(f"[✓] Index saved to {INDEX_SAVE_PATH}")

# === Run Script ===
if __name__ == "__main__":
    print("[*] Loading documents...")
    docs = load_documents()
    print(f"[*] Loaded {len(docs)} documents")

    print("[*] Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"[*] Split into {len(chunks)} chunks")

    print("[*] Building FAISS index with all-mpnet-base-v2...")
    build_faiss_index(chunks, EMBEDDING_MODEL_NAME)

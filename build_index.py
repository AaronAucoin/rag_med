import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Placeholder medical knowledge base
documents = [
    "Acetaminophen is used to relieve mild to moderate pain and reduce fever.",
    "Ibuprofen is commonly used to reduce inflammation, pain, and fever.",
    "Amoxicillin is prescribed for bacterial infections like strep throat.",
    "Diphenhydramine is used for allergies, hay fever, and the common cold.",
    "Loperamide is used to control diarrhea."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

with open("rag_index.pkl", "wb") as f:
    pickle.dump((index, documents), f)

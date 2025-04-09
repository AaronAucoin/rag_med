import faiss
import pickle
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import threading

app = Flask(__name__)

# Global variables
generator = None
model_ready = False

# Load FAISS index and documents
with open("rag_index.pkl", "rb") as f:
    index, documents = pickle.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_model():
    global generator, model_ready
    print("Loading the model...")
    generator = pipeline("text-generation", model="distilgpt2")
    model_ready = True
    print("Model loaded successfully!")

# Load the model in a separate thread to prevent blocking the Flask server
thread = threading.Thread(target=load_model)
thread.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if not model_ready:
        return jsonify({"error": "Model is still loading, please try again later."}), 503

    data = request.get_json()
    symptoms = data.get('symptoms')

    # Step 1: Embed user query
    query_embedding = embedder.encode([symptoms])

    # Step 2: Retrieve top document
    D, I = index.search(query_embedding, k=1)
    context = documents[I[0][0]]

    # Step 3: Construct prompt
    prompt = f"Symptoms: {symptoms}\nRelevant Info: {context}\n\nRecommend a medicine and explain why."

    # Step 4: Generate response
    raw_output = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

    # Simple post-processing (very basic)
    med = raw_output.split("Recommend a medicine and explain why.")[-1].strip().split('.')[0]
    reason = '.'.join(raw_output.split('.')[1:]).strip()

    return jsonify({
        'medicine': med,
        'reason': reason or "Explanation not available."
    })

if __name__ == '__main__':
    app.run(debug=True)

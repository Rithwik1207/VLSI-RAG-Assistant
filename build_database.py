import os
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SECURITY ---
# Load environment variables from the .env file to protect your API key
load_dotenv()

# Local relative path to your textbook
file_path = './data/VLSI Design_Neil Weste_Text book.pdf'

if not os.path.exists(file_path):
    raise FileNotFoundError("Could not find the PDF. Ensure it is in the 'data' folder.")

# --- 2. DATA EXTRACTION & PROCESSING ---
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def shred_into_chunks(text, size=512, overlap=50):
    chunks = []
    current_position = 0
    while current_position < len(text):
        chunk_end = current_position + size
        chunks.append(text[current_position : chunk_end])
        current_position += (size - overlap)
    return chunks

print("Extracting and shredding textbook...")
raw_book_text = extract_text_from_pdf(file_path)
text_chunks = shred_into_chunks(raw_book_text)
print(f"Total chunks created: {len(text_chunks)}")

# --- 3. VECTOR DATABASE (FAISS) CONSTRUCTION ---
print("Loading local embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Translating text into math vectors...")
embeddings = embedding_model.encode(text_chunks)
dimension = embeddings.shape[1]

# Create and populate the FAISS Index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
print(f"Index ready! Total Vectors stored: {index.ntotal}")

# --- 4. EXPORTING THE BRAIN ---
print("Saving vector database and text chunks locally...")
# These will save directly to your current VS Code folder
faiss.write_index(index, "vector_index.faiss")
with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("Export complete. You are ready to run the Streamlit dashboard!")
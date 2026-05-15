# 🔌 VLSI Engineering RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** application designed to assist engineering students and professionals with complex VLSI (Very Large Scale Integration) design queries. This assistant uses a vectorized version of the Neil Weste VLSI textbook to provide accurate, context-aware answers.

## 🚀 Live Demo
https://vlsi-rag-assistant.streamlit.app/


## ✨ Key Features
* **Contextual Intelligence:** Uses RAG to answer questions strictly based on textbook data, reducing AI "hallucinations."
* **Mathematical Precision:** Automatically formats formulas using LaTeX for clear engineering communication ($E=mc^2$ style).
* **Conversation Memory:** Built-in session state management allows the assistant to remember previous questions in a thread.
* **Vector Search:** Utilizes FAISS (Facebook AI Similarity Search) for millisecond retrieval of relevant textbook excerpts.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **LLM API:** Groq (Llama 3.1 8B)
* **Vector Database:** FAISS (FlatL2 Index)
* **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
* **Frontend:** Streamlit
* **PDF Processing:** PyMuPDF (fitz)

## 🏗️ Architecture
1.  **Ingestion (`build_database.py`):** Extracts text from the VLSI textbook, shreds it into overlapping chunks, and converts it into mathematical vectors.
2.  **Storage:** Saves the vectors into a `.faiss` index and text into a `.pkl` file for lightweight deployment.
3.  **Retrieval & Generation (`app.py`):** * User asks a question.
    * System finds the top 3 most relevant textbook chunks.
    * Groq LLM synthesizes a final answer using the chunks as evidence.

## 💻 Local Setup

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Rithwik1207/VLSI-RAG-Assistant.git](https://github.com/Rithwik1207/VLSI-RAG-Assistant.git)
   cd VLSI-RAG-Assistant

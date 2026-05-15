import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# --- 1. System Initialization ---
load_dotenv()
st.set_page_config(page_title="VLSI AI Assistant", layout="centered")
st.title("🔌 VLSI Engineering Knowledge Base")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# --- 2. Initialize Memory (The Notepad) ---
# If this is the first time opening the app, create an empty list for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Load the Exported Data ---
@st.cache_resource
def load_backend():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("vector_index.faiss")
    with open("text_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return embedding_model, index, chunks

embedding_model, index, text_chunks = load_backend()

# --- 4. The User Interface ---

# Step A: Draw all the PAST messages to the screen so it looks like a real chat app
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Step B: Wait for a NEW question
user_question = st.chat_input("Ask a question about VLSI design...")

if user_question:
    # Immediately draw the user's new question on the screen
    with st.chat_message("user"):
        st.write(user_question)

    with st.spinner("Searching textbook vectors..."):
        # [Retrieval Phase]
        question_vector = embedding_model.encode([user_question]).astype('float32')
        distances, indices = index.search(question_vector, k=3)
        
        context_text = ""
        for i in range(len(indices[0])):
            context_text += f"Excerpt {i+1}:\n{text_chunks[indices[0][i]]}\n\n"
        
        # [Generation Phase - Building the Movie Script]
        # 1. Start with the strict rules
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional VLSI Engineering Assistant. "
                    "You answer questions based STRICTLY on the provided textbook excerpts. "
                    "If the answer is not in the excerpts, explicitly say 'I cannot find this in the textbook.' "
                    "Fix any broken words from the excerpts. Format all math formulas in LaTeX using $ for inline and $$ for block equations."
                )
            }
        ]
        
        # 2. Append the PAST conversation history (so the AI remembers what was just said)
        # We limit it to the last 4 messages so we don't overwhelm the AI's token limit
        for msg in st.session_state.chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # 3. Finally, append the CURRENT question and the fresh textbook chunks
        messages.append({
            "role": "user",
            "content": f"TEXTBOOK EXCERPTS:\n{context_text}\n\nUSER QUESTION: {user_question}"
        })
        
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.2,
            )
            answer = chat_completion.choices[0].message.content
            
            # Draw the AI's new answer on the screen
            with st.chat_message("assistant"):
                st.markdown(answer)
                
            # Step C: Save this interaction to our Memory Notepad for next time!
            # Notice we only save the clean user question, not the giant textbook chunks, to keep memory efficient
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"API Error: {e}")
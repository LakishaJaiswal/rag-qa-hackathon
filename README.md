# 🤖 RAG-based Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** system using local text documents and a language model to answer user questions. It combines the power of **semantic search** with **generative reasoning** to provide accurate and context-aware answers grounded in your custom knowledge base.

---

## 📚 Use Case: Cybersecurity Awareness

This project is tailored for cybersecurity education. It supports answering questions on topics such as:

- Phishing
- Malware
- Encryption
- VPNs
- Cyber attacks
- Online safety
- Digital hygiene

You can add more `.txt` files in the `docs/` folder for expanding the knowledge base.

---

## 💡 How It Works

1. **Documents Loader:** Loads and chunks `.txt` documents.
2. **Embedding & Indexing:** Converts chunks to embeddings using `sentence-transformers` and stores them in a FAISS index.
3. **Retriever:** Finds top relevant chunks using similarity search.
4. **Generator:** Uses a language model (`flan-t5-base`) to generate an answer based on the context from relevant documents.

---

## 🧠 Tech Stack

- Python 3.10+
- HuggingFace Transformers
- SentenceTransformers
- FAISS
- LangChain (optional for expansion)
- Gradio (optional for web UI)

---

## 📁 Project Structure


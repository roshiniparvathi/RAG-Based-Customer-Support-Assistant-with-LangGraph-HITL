# RAG-Based-Customer-Support-Assistant-with-LangGraph-HITL
# 🚀 RAG Customer Support Assistant

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) based customer support assistant built using LangGraph and Groq.

It processes a PDF knowledge base and generates context-aware answers with citation support.

---

## ⚙️ Features

* PDF ingestion and chunking
* Embedding-based retrieval (ChromaDB)
* Context-aware response generation (Groq LLM)
* LangGraph workflow orchestration
* Human-in-the-Loop (HITL) escalation
* Streamlit UI

---

## 🧠 Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* HuggingFace Embeddings
* Groq API
* Streamlit

---

## 🔄 Workflow

1. Upload PDF
2. Chunk + embed
3. Store in ChromaDB
4. Query → retrieve → generate answer
5. Route → Answer / HITL

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

Set API Key:

```bash
export GROQ_API_KEY=your_key
```

Run:

```bash
streamlit run streamlit_app.py
```

---

## 📚 Documents

* HLD
* LLD
* Technical Documentation

---

## 🎥 Demo

[Paste Google Drive link here]

---

## 📈 Future Improvements

* Multi-document support
* Memory-based conversations
* API deployment

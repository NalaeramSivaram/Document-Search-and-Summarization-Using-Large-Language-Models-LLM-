import streamlit as st

st.set_page_config(
    page_title="Document Search & Summarization",
    layout="centered"
)

st.title("📄 Document Search & Question Answering System")

st.markdown("""
## 🔍 Overview
This application allows users to upload documents (PDF or TXT) and ask
natural language questions about their content.

The system returns **precise, sentence-level answers** along with a
**reliable confidence score** that reflects how well the answer is grounded
in the document.

---

## ⚙️ How the System Works

### 1️⃣ Document Processing
- Uploaded documents are cleaned and split into chunks.
- Text chunks are indexed using semantic embeddings.

### 2️⃣ Hybrid Retrieval
- Semantic search (FAISS)
- Keyword-aware filtering (prevents concept drift)

### 3️⃣ Precise Question Answering
- Answers are extracted at the **sentence level**
- Section headings are automatically expanded into explanations
- Paragraphs are returned **only for summaries**

### 4️⃣ Confidence Score
- Calculated using **question + answer + document context**
- Prevents false high-confidence wrong answers

---

## ▶️ How to Use

1. Go to **Document Q&A** from the sidebar  
2. Upload a document  
3. Ask questions like:
   - *What is deep learning?*
   - *Explain supervised learning*
   - *Summarize the document*

---

## 🎯 Key Features
- Accurate answers
- Honest confidence scoring
- Free & offline
- Interview-ready design

➡️ **Use the sidebar to start.**
""")

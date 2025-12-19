import streamlit as st
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize
from sentence_transformers import util

from preprocessing import process_uploaded_file
from embedding_index import EmbeddingIndex
from search_engine import HybridSearch

from collections import Counter
import math

st.set_page_config(page_title="Document Q&A")
st.title("📄 Document Question Answering")

# ---------- Background Image ----------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.shutterstock.com/image-illustration/two-books-covered-word-question-260nw-312479006.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Optional: make content readable */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------
# Session state
# ----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ----------------------------
# Helper functions
# ----------------------------
def is_summary_query(q):
    return any(w in q.lower() for w in
               ["summary", "summarize", "overview", "entire", "brief"])

def keyword_overlap(question, sentence):
    return len(set(question.lower().split())
               .intersection(set(sentence.lower().split())))

def is_heading(sentence):
    return sentence.strip().endswith("?") or len(sentence.split()) < 6

def get_precise_answer(question, chunks, embedder):
    sentences = []
    for c in chunks:
        sentences.extend(sent_tokenize(c))

    q_emb = embedder.model.encode(question, convert_to_tensor=True)
    s_emb = embedder.model.encode(sentences, convert_to_tensor=True)

    semantic_scores = util.cos_sim(q_emb, s_emb)[0]

    best_score, best_idx = -1, 0
    for i, sent in enumerate(sentences):
        score = semantic_scores[i] + (0.15 * keyword_overlap(question, sent))
        if score > best_score:
            best_score, best_idx = score, i

    best_sentence = sentences[best_idx]

    # Expand heading into explanation
    if is_heading(best_sentence) and best_idx + 1 < len(sentences):
        best_sentence += " " + sentences[best_idx + 1]

    return best_sentence, sentences

def calculate_text_accuracy(answer, question, sentences, embedder):
    combined = question + " " + answer
    comb_emb = embedder.model.encode(combined, convert_to_tensor=True)
    sent_emb = embedder.model.encode(sentences, convert_to_tensor=True)
    return float(util.cos_sim(comb_emb, sent_emb).max())

def clean_answer_text(answer):
    prefixes = [
        "Definition :",
        "Definition:",
        "What is",
        "What Is",
        "WHAT IS"
    ]

    for p in prefixes:
        if p in answer:
            answer = answer.split(p, 1)[-1].strip()

    return answer

def summarize_document(chunks, embedder, max_sentences=8):
    sentences = []
    for c in chunks:
        sentences.extend(sent_tokenize(c))

    # Remove very short or noisy sentences
    sentences = [s for s in sentences if len(s.split()) > 6]

    # Compute embeddings
    sent_emb = embedder.model.encode(sentences, convert_to_tensor=True)
    doc_emb = sent_emb.mean(dim=0)

    # Semantic importance
    semantic_scores = util.cos_sim(doc_emb, sent_emb)[0]

    # Keyword importance
    words = " ".join(sentences).lower().split()
    freq = Counter(words)

    keyword_scores = []
    for s in sentences:
        score = sum(freq[w] for w in s.lower().split())
        keyword_scores.append(score)

    # Normalize keyword scores
    max_kw = max(keyword_scores)
    keyword_scores = [k / max_kw for k in keyword_scores]

    # Final score
    final_scores = [
        float(semantic_scores[i]) + 0.3 * keyword_scores[i]
        for i in range(len(sentences))
    ]

    # Select top sentences
    ranked = sorted(
        zip(sentences, final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    summary = [s for s, _ in ranked[:max_sentences]]

    return " ".join(summary)
# ----------------------------
# Upload documents
# ----------------------------
files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if files and st.session_state.chunks is None:
    chunks = []
    with st.spinner("Processing documents..."):
        for f in files:
            chunks.extend(process_uploaded_file(f))

        embedder = EmbeddingIndex()
        embedder.build_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.embedder = embedder
    st.success(f"Indexed {len(chunks)} text chunks")

# ----------------------------
# Ask question
# ----------------------------
if st.session_state.chunks:
    query = st.text_input("Ask a question (e.g., What is deep learning?)")

    if st.button("Get Answer"):
        embedder = st.session_state.embedder
        search_engine = HybridSearch(st.session_state.chunks)

        semantic = embedder.search(query)
        relevant = search_engine.search(query, semantic)

        if not is_summary_query(query):
            answer, sentences = get_precise_answer(query, relevant, embedder)

            if keyword_overlap(query, answer) == 0:
                st.error("❌ The document does not clearly contain an answer.")
                st.stop()

            accuracy = calculate_text_accuracy(
                answer, query, sentences, embedder
            )

            answer = clean_answer_text(answer)

            st.subheader("✅ Answer")
            st.write(answer)
            st.caption(f"📊 Answer Confidence Score: **{accuracy:.2f}**")

            if accuracy >= 0.8:
                st.success("High confidence answer")
            elif accuracy >= 0.6:
                st.info("Acceptable confidence answer")
            else:
                st.warning("Low confidence answer")

        else:
            st.subheader("📝 Summary")

            summary = summarize_document(
                relevant,
                st.session_state.embedder,
                max_sentences=8
                )

            st.write(summary)

else:
    st.info("Please upload documents to begin.")

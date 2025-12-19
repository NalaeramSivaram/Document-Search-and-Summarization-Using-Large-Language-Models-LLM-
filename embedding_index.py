import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingIndex:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_index(self, docs):
        self.docs = docs
        emb = np.array(self.model.encode(docs)).astype("float32")
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(emb)

    def search(self, query, top_k=3):
        q = self.model.encode([query]).astype("float32")
        _, idx = self.index.search(q, top_k)
        return [self.docs[i] for i in idx[0]]

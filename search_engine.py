from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridSearch:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query, semantic_docs, top_k=3):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        tfidf_docs = [self.docs[i] for i in scores.argsort()[-top_k:]]
        return list(dict.fromkeys(semantic_docs + tfidf_docs))

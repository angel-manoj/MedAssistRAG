import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.index = faiss.read_index("data/embeddings/faiss_index.bin")

        with open("data/embeddings/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        self.model = SentenceTransformer("BAAI/bge-small-en")

    def search(self, query, k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb, k)

        return [self.texts[i] for i in I[0]]
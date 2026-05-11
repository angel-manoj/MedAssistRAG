import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "qa_dataset_large.csv"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
EMBED_DATA_PATH = BASE_DIR / "data" / "embedded_data.csv"


class Retriever:
    def __init__(self):
        print("[Retriever] Loading Sentence Transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Check if FAISS index already exists
        if INDEX_PATH.exists() and EMBED_DATA_PATH.exists():
            print("[Retriever] Loading existing FAISS index...")
            self.index = faiss.read_index(str(INDEX_PATH))
            self.df = pd.read_csv(EMBED_DATA_PATH)
        else:
            print("[Retriever] Building FAISS index from data...")
            self._build_index()

    def _build_index(self):
        """Load CSV data, embed questions, and build FAISS index."""
        self.df = pd.read_csv(DATA_PATH)

        # Drop rows with missing question or answer
        self.df = self.df.dropna(subset=["question", "answer"]).reset_index(drop=True)

        print(f"[Retriever] Loaded {len(self.df)} QA pairs")

        # Encode questions using sentence transformer
        print("[Retriever] Encoding questions (this may take a few minutes)...")
        embeddings = self.model.encode(
            self.df["question"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype="float32")

        # Build FAISS index (Inner Product for cosine similarity on normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save index and data for reuse
        faiss.write_index(self.index, str(INDEX_PATH))
        self.df.to_csv(EMBED_DATA_PATH, index=False)

        print(f"[Retriever] FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query, top_k=5):
        """Search for the most relevant QA pairs given a query."""
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = self.df.iloc[indices[0]].copy()
        results["score"] = scores[0]

        return results

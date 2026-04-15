from sentence_transformers import SentenceTransformer

def load_model():
    model = SentenceTransformer("BAAI/bge-small-en")
    return model

def create_embeddings(model, texts):
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings
from sentence_transformers import SentenceTransformer


_MODEL_CACHE = {}


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

        if model_name not in _MODEL_CACHE:
            print(f"Loading embedding model: {model_name}")
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

        self.model = _MODEL_CACHE[model_name]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
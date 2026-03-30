from sentence_transformers import CrossEncoder


_RERANKER_CACHE = {}


class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

        if model_name not in _RERANKER_CACHE:
            print(f"Loading reranker model: {model_name}")
            _RERANKER_CACHE[model_name] = CrossEncoder(model_name)

        self.model = _RERANKER_CACHE[model_name]

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = []
        for item in candidates:
            raw_text = item["metadata"].get("raw_text", "")
            section_title = item["metadata"].get("section_title", "")
            source_name = item["metadata"].get("source_name", "")

            candidate_text = (
                f"Source: {source_name}\n"
                f"Section: {section_title}\n"
                f"Content: {raw_text}"
            )
            pairs.append((query, candidate_text))

        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            updated = dict(item)
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
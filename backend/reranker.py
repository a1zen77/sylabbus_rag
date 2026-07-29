import os
from sentence_transformers import CrossEncoder

# Small, fast cross-encoder well-suited for CPU inference.
# Trained on MS MARCO (a large passage-ranking dataset) — good general-purpose reranker.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load once at module import time (singleton pattern, same as bm25_index)
print(f"Loading cross-encoder model: {MODEL_NAME}...")
_reranker_model = CrossEncoder(MODEL_NAME)
print("Cross-encoder loaded.")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank a list of candidate chunks using a cross-encoder.

    candidates: list of dicts with at least a "text" key (output of hybrid_retrieve
                or vector_search works directly).
    Returns: top_k candidates, re-sorted by cross-encoder relevance score,
             with a "rerank_score" field added.
    """
    if not candidates:
        return []

    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, c["text"]) for c in candidates]

    scores = _reranker_model.predict(pairs)

    # Attach scores and sort descending (higher = more relevant)
    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["rerank_score"] = float(score)
        scored_candidates.append(item)

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_candidates[:top_k]


if __name__ == "__main__":
    # Quick standalone test
    test_query = "What is the course code for NLP?"
    test_candidates = [
        {"text": "The course code for Natural Language Processing is 410252."},
        {"text": "Deep Learning (410251) carries 3 credits."},
        {"text": "Prerequisites include Data Structures and Algorithms."},
    ]

    results = rerank(test_query, test_candidates, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r['rerank_score']:.3f}] {r['text']}")
import os
import re
import chromadb
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name="sppu_syllabi")


def simple_tokenize(text: str) -> list[str]:
    """
    Basic tokenizer: lowercase, split on non-alphanumeric characters.
    Good enough for BM25 — no need for a heavy NLP tokenizer here.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Index:
    """
    Wraps a BM25 index built from all chunks currently in ChromaDB.
    Rebuilt on-demand (call .refresh()) whenever documents are added/deleted,
    since BM25Okapi doesn't support incremental updates.
    """
    def __init__(self):
        self.bm25 = None
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.refresh()

    def refresh(self):
        """Rebuild the BM25 index from current ChromaDB contents."""
        all_data = collection.get()

        self.ids = all_data['ids']
        self.documents = all_data['documents']
        self.metadatas = all_data['metadatas']

        if not self.documents:
            self.bm25 = None
            return

        tokenized_corpus = [simple_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks ranked by BM25 score."""
        if self.bm25 is None:
            return []

        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top_k highest scores
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "bm25_score": scores[idx]
            })

        return results

# Singleton instance — built once when this module is imported
bm25_index = BM25Index()

def embed_query(query: str) -> list[float]:
    """Embed the user's question (same as retriever.py)."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']


def vector_search(query: str, top_k: int = 10) -> list[dict]:
    """Dense vector search via ChromaDB."""
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted = []
    for i in range(len(results['ids'][0])):
        formatted.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    return formatted


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    Returns a single re-ranked list, deduplicated by chunk id.
    """
    scores = {}
    chunk_data = {}

    # Score contribution from vector search ranking
    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = item

    # Score contribution from BM25 ranking
    for rank, item in enumerate(bm25_results, start=1):
        chunk_id = item["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = item

    # Sort by fused score, descending
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results = []
    for chunk_id in ranked_ids:
        item = chunk_data[chunk_id].copy()
        item["rrf_score"] = scores[chunk_id]
        fused_results.append(item)

    return fused_results


def hybrid_retrieve(query: str, top_k: int = 5, candidate_pool: int = 10) -> list[dict]:
    """
    Main entry point: run vector + BM25 search, fuse rankings, return top_k.

    candidate_pool: how many results to pull from EACH method before fusion
                    (larger pool = fusion has more to work with, but slower).
    """
    vector_results = vector_search(query, top_k=candidate_pool)
    bm25_results = bm25_index.search(query, top_k=candidate_pool)

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    return fused[:top_k]


if __name__ == "__main__":
    test_query = "What is the course code for NLP?"
    print(f"Question: {test_query}\n")

    print("=== Vector-only results ===")
    for i, r in enumerate(vector_search(test_query, top_k=5), 1):
        print(f"{i}. [dist={r['distance']:.3f}] {r['text'][:100]}...")

    print("\n=== BM25-only results ===")
    for i, r in enumerate(bm25_index.search(test_query, top_k=5), 1):
        print(f"{i}. [score={r['bm25_score']:.3f}] {r['text'][:100]}...")

    print("\n=== Hybrid (RRF fused) results ===")
    for i, r in enumerate(hybrid_retrieve(test_query, top_k=5), 1):
        print(f"{i}. [rrf={r['rrf_score']:.4f}] {r['text'][:100]}...")
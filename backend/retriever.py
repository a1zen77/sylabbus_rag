import os
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

from hybrid_retriever import hybrid_retrieve

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Chroma
# Resolve path relative to this file's location, not the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Get the collection
collection = chroma_client.get_collection(name="sppu_syllabi")

def embed_query(query: str) -> list[float]:
    """Embed the user's question."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Search Chroma for relevant chunks."""
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    contexts = []
    for i in range(len(results['ids'][0])):
        contexts.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i] if 'distances' in results else None
        })
    
    return contexts

def check_confidence(contexts: list[dict], threshold: float = 0.5) -> bool:
    """
    Check if the best result meets the confidence threshold.
    Returns True if confident, False if not.
    """
    if not contexts:
        return False
    
    # Get the best (lowest) distance
    best_distance = contexts[0].get("distance", 1.0)
    
    # If distance is greater than threshold, we're not confident
    return best_distance <= threshold

def retrieve_context_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval (vector + BM25 fused via RRF).
    Same output shape as retrieve_context() so it's a drop-in replacement.
    """
    results = hybrid_retrieve(query, top_k=top_k)

    # Normalize shape to match retrieve_context()'s output
    contexts = []
    for r in results:
        contexts.append({
            "text": r["text"],
            "metadata": r["metadata"],
            "distance": r.get("distance", 1 - r.get("rrf_score", 0))  # approximate for confidence check
        })
    return contexts

if __name__ == "__main__":
    test_queries = [
        "What are the course objectives?",  # Should be confident
        "What's the professor's email?"      # Should NOT be confident
    ]
    
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("="*60)
        
        contexts = retrieve_context(query, top_k=5)
        is_confident = check_confidence(contexts, threshold)
        
        print(f"Best match distance: {contexts[0]['distance']:.4f}")
        print(f"Threshold: {threshold}")
        print(f"Confident enough to answer? {'YES ✅' if is_confident else 'NO ❌'}")
        
        if contexts:
            print(f"\nTop result: {contexts[0]['text'][:150]}...")
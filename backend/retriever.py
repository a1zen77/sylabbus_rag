import os
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Chroma
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
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
    # Convert question to embedding
    query_embedding = embed_query(query)
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    contexts = []
    for i in range(len(results['ids'][0])):
        contexts.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i] if 'distances' in results else None
        })
    
    return contexts

if __name__ == "__main__":
    # Test the retriever
    test_query = "What are the course objectives?"
    print(f"Question: {test_query}\n")
    
    results = retrieve_context(test_query)
    
    print(f"Found {len(results)} relevant chunks:\n")
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {result['metadata']['filename']}, Chunk: {result['metadata']['chunk_index']}")
        if result['distance']:
            print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print()
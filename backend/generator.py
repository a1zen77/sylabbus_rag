import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def truncate_context(contexts: list[dict], max_chars_per_chunk: int = 500) -> list[dict]:
    """Truncate each chunk to save tokens."""
    truncated = []
    for ctx in contexts:
        truncated.append({
            "text": ctx["text"][:max_chars_per_chunk] + ("..." if len(ctx["text"]) > max_chars_per_chunk else ""),
            "metadata": ctx["metadata"],
            "distance": ctx.get("distance")
        })
    return truncated

def build_prompt(query: str, contexts: list[dict]) -> str:
    """Build RAG prompt with context."""
    context_text = "\n\n".join([
        f"[{ctx['metadata']['filename']}, Chunk {ctx['metadata']['chunk_index']}]\n{ctx['text']}"
        for ctx in contexts
    ])
    
    prompt = f"""Answer using only the context below. If info is missing, say "I don't have enough information."

Context:
{context_text}

Question: {query}

Answer:"""
    
    return prompt

def generate_answer(query: str, contexts: list[dict], max_retries: int = 3) -> str:
    """Generate answer using Gemini with retry logic."""
    contexts = truncate_context(contexts, max_chars_per_chunk=500)
    prompt = build_prompt(query, contexts)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                wait_time = (attempt + 1) * 5
                print(f"⏳ Rate limit hit. Waiting {wait_time}s... (retry {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    
    return "Error: Rate limit exceeded. Please try again later."

if __name__ == "__main__":
    from retriever import retrieve_context
    
    test_query = "What are the course objectives?"
    print(f"Question: {test_query}\n")
    
    contexts = retrieve_context(test_query, top_k=3)  # Reduced from 5
    print(f"Found {len(contexts)} chunks\n")
    
    answer = generate_answer(test_query, contexts)
    print(f"\nAnswer:\n{answer}")
import os
import ollama
from dotenv import load_dotenv

load_dotenv()

def truncate_context(contexts: list[dict], max_chars_per_chunk: int = 600) -> list[dict]:
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
    
    prompt = f"""Answer the question using ONLY the context below. Be concise and precise.

Context:
{context_text}

Question: {query}

If the context doesn't contain the answer, respond with: "I don't have enough information."

Answer:"""
    
    return prompt

def generate_answer(query: str, contexts: list[dict]) -> str:
    """Generate answer using Ollama (non-streaming)."""
    contexts = truncate_context(contexts, max_chars_per_chunk=600)
    prompt = build_prompt(query, contexts)
    
    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 200,
            }
        )
        
        return response['message']['content']
    
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return f"Error generating answer: {str(e)}"

def generate_answer_stream(query: str, contexts: list[dict]):
    """Generate answer using Ollama with streaming."""
    contexts = truncate_context(contexts, max_chars_per_chunk=600)
    prompt = build_prompt(query, contexts)
    
    try:
        stream = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 200,
            },
            stream=True  # ← Enable streaming
        )
        
        # Yield each chunk as it arrives
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    except Exception as e:
        yield f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    from retriever import retrieve_context
    
    test_query = "What are the course objectives?"
    print(f"Question: {test_query}\n")
    
    print("Retrieving relevant chunks...")
    contexts = retrieve_context(test_query, top_k=3)
    print(f"Found {len(contexts)} chunks\n")
    
    print("Generating answer with streaming...")
    print("Answer: ", end="", flush=True)
    
    for chunk in generate_answer_stream(test_query, contexts):
        print(chunk, end="", flush=True)
    
    print("\n")
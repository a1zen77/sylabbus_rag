import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the generative model
model = genai.GenerativeModel('gemini-2.0-flash')

def build_prompt(query: str, contexts: list[dict]) -> str:
    """Build RAG prompt with context."""
    # Format each context chunk
    context_text = "\n\n".join([
        f"[Source: {ctx['metadata']['filename']}, Chunk {ctx['metadata']['chunk_index']}]\n{ctx['text']}"
        for ctx in contexts
    ])
    
    # Build the full prompt
    prompt = f"""You are a helpful assistant answering questions about SPPU university syllabi.

Context from syllabus documents:
{context_text}

Question: {query}

Instructions:
- Answer based ONLY on the context provided above
- Be precise and concise
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided syllabus to answer this question."
- Do not make up information or use knowledge outside the provided context

Answer:"""
    
    return prompt

def generate_answer(query: str, contexts: list[dict]) -> str:
    """Generate answer using Gemini."""
    # Build prompt with context
    prompt = build_prompt(query, contexts)
    
    # Generate response
    response = model.generate_content(prompt)
    
    return response.text

if __name__ == "__main__":
    from retriever import retrieve_context
    
    test_query = "What are the course objectives?"
    print(f"Question: {test_query}\n")
    
    # Retrieve fewer chunks
    contexts = retrieve_context(test_query, top_k=3)  # ← Changed from 5 to 3
    print(f"Found {len(contexts)} chunks\n")
    
    answer = generate_answer(test_query, contexts)
    print(f"\nAnswer:\n{answer}")
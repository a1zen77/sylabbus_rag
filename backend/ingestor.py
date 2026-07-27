import os
import fitz  # PyMuPDF
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Chroma with persistent client
# Resolve path relative to this file's location, not the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="sppu_syllabi",
    metadata={"hnsw:space": "cosine"}
)

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF, preserving page numbers."""
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():  # Only add non-empty pages
            pages_data.append({
                "page_number": page_num,
                "text": text
            })
    
    doc.close()
    return pages_data

def chunk_text_with_pages(pages_data: list[dict], chunk_size: int = 800) -> list[dict]:
    """Split text into chunks while preserving page numbers."""
    chunks = []
    
    for page_data in pages_data:
        page_num = page_data["page_number"]
        text = page_data["text"]
        words = text.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "page_number": page_num
                })
                current_chunk = []
                current_size = 0
        
        # Add remaining words from this page
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "page_number": page_num
            })
    
    return chunks

def embed_text(text: str) -> list[float]:
    """Generate embedding using Gemini."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

def ingest_pdf(pdf_path: str, filename: str):
    """Main ingestion pipeline with page tracking."""
    print(f"Processing {filename}...")
    
    # Extract text with page numbers
    pages_data = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {len(pages_data)} pages")
    
    # Chunk text while preserving page numbers
    chunks = chunk_text_with_pages(pages_data, int(os.getenv("CHUNK_SIZE", 800)))
    print(f"Created {len(chunks)} chunks")
    
    # Embed and store each chunk
    for i, chunk_data in enumerate(chunks):
        embedding = embed_text(chunk_data["text"])
        
        collection.add(
            ids=[f"{filename}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk_data["text"]],
            metadatas=[{
                "filename": filename,
                "chunk_index": i,
                "page_number": chunk_data["page_number"]  # ← NEW: Store page number
            }]
        )
    
    print(f"✓ Ingested {filename}")
    print(f"✓ Data persisted to {CHROMA_DIR}")
    return len(chunks)

if __name__ == "__main__":
    # Test ingestion
    test_pdf = "./data/syllabi/sample.pdf"
    if os.path.exists(test_pdf):
        ingest_pdf(test_pdf, "sample.pdf")
    else:
        print(f"Error: {test_pdf} not found")
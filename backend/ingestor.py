import os
import fitz  # PyMuPDF
import google.generativeai as genai
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Chroma
chroma_client = Client(Settings(
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    anonymized_telemetry=False
))

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="sppu_syllabi",
    metadata={"hnsw:space": "cosine"}
)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 800) -> list[str]:
    """Split text into chunks of roughly equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # Add remaining words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
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
    """Main ingestion pipeline."""
    print(f"Processing {filename}...")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Chunk text
    chunks = chunk_text(text, int(os.getenv("CHUNK_SIZE", 800)))
    print(f"Created {len(chunks)} chunks")
    
    # Embed and store each chunk
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        
        collection.add(
            ids=[f"{filename}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"filename": filename, "chunk_index": i}]
        )
    
    print(f"✓ Ingested {filename}")
    return len(chunks)

# Add this at the bottom temporarily
if __name__ == "__main__":
    # Download a sample SPPU syllabus PDF or use any PDF
    test_pdf = "./data/syllabi/sample.pdf"
    ingest_pdf(test_pdf, "sample.pdf")
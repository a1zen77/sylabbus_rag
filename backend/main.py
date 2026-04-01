import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestor import ingest_pdf
from retriever import retrieve_context
from generator import generate_answer

load_dotenv()

app = FastAPI(title="SPPU Syllabus QA API")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

# Routes
@app.get("/")
async def root():
    return {
        "message": "SPPU Syllabus QA API is running",
        "endpoints": {
            "POST /ingest": "Upload and ingest a PDF",
            "POST /chat": "Ask a question and get an answer"
        }
    }

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF syllabus."""
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create upload directory
    upload_dir = os.getenv("UPLOAD_DIR", "./data/syllabi")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Ingest the PDF
    try:
        num_chunks = ingest_pdf(file_path, file.filename)
    except Exception as e:
        # Clean up file if ingestion fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {str(e)}")
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks": num_chunks,
        "message": f"Successfully ingested {file.filename} with {num_chunks} chunks"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question and get an answer with sources."""
    try:
        # Retrieve relevant context
        top_k = int(os.getenv("TOP_K", 5))
        contexts = retrieve_context(request.question, top_k=top_k)
        
        if not contexts:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found. Please upload syllabus PDFs first."
            )
        
        # Generate answer
        answer = generate_answer(request.question, contexts)
        
        # Format sources (now with page numbers)
        sources = [
            {
                "filename": ctx["metadata"]["filename"],
                "page_number": ctx["metadata"]["page_number"],  # ← NEW
                "chunk_index": ctx["metadata"]["chunk_index"],
                "text_preview": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "distance": ctx.get("distance")
            }
            for ctx in contexts
        ]
        
        return ChatResponse(answer=answer, sources=sources)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
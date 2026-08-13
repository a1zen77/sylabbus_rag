import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json

from ingestor import ingest_pdf
from retriever import retrieve_context, check_confidence  # ← Import check_confidence
from generator import generate_answer, generate_answer_stream
from extractor import extract_course_info

load_dotenv()

app = FastAPI(title="SPPU Syllabus QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confident: bool  # ← NEW: Indicate if we're confident

class ExtractRequest(BaseModel):
    course_query: str

@app.get("/")
async def root():
    return {
        "message": "SPPU Syllabus QA API is running",
        "endpoints": {
            "POST /ingest": "Upload and ingest a PDF",
            "POST /chat": "Ask a question and get an answer",
            "POST /chat/stream": "Ask a question and stream the answer",
            "GET /documents": "List all indexed documents",
            "DELETE /documents/{filename}": "Delete a document from the index",
            "POST /extract": "Extract structured course info (credits, exam pattern)"
        }
    }

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF syllabus."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    upload_dir = os.getenv("UPLOAD_DIR", "./data/syllabi")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        num_chunks = ingest_pdf(file_path, file.filename)
    except Exception as e:
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
    """Ask a question and get an answer with sources (non-streaming)."""
    try:
        top_k = int(os.getenv("TOP_K", 5))
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        
        contexts = retrieve_context(request.question, top_k=top_k)
        
        if not contexts:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found. Please upload syllabus PDFs first."
            )
        
        # Check confidence
        is_confident = check_confidence(contexts, threshold)
        
        # Generate answer (or decline if not confident)
        if is_confident:
            answer = generate_answer(request.question, contexts)
        else:
            answer = f"I don't have enough information in the provided syllabus to answer this question confidently. The best match I found has a relevance of {(1 - contexts[0]['distance']):.1%}, which is below my confidence threshold of {(1 - threshold):.1%}."
        
        sources = [
            {
                "filename": ctx["metadata"]["filename"],
                "page_number": ctx["metadata"]["page_number"],
                "chunk_index": ctx["metadata"]["chunk_index"],
                "text_preview": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "distance": ctx.get("distance")
            }
            for ctx in contexts
        ]
        
        return ChatResponse(answer=answer, sources=sources, confident=is_confident)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Ask a question and stream the answer with sources."""
    try:
        top_k = int(os.getenv("TOP_K", 5))
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        
        contexts = retrieve_context(request.question, top_k=top_k)
        
        if not contexts:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload syllabus PDFs first."
            )
        
        # Check confidence
        is_confident = check_confidence(contexts, threshold)
        
        sources = [
            {
                "filename": ctx["metadata"]["filename"],
                "page_number": ctx["metadata"]["page_number"],
                "chunk_index": ctx["metadata"]["chunk_index"],
                "text_preview": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "distance": ctx.get("distance")
            }
            for ctx in contexts
        ]
        
        async def generate():
            # First, send sources and confidence
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'confident': is_confident})}\n\n"
            
            # Then stream the answer
            if is_confident:
                for chunk in generate_answer_stream(request.question, contexts):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            else:
                low_confidence_msg = f"I don't have enough information in the provided syllabus to answer this question confidently. The best match I found has a relevance of {(1 - contexts[0]['distance']):.1%}, which is below my confidence threshold of {(1 - threshold):.1%}."
                yield f"data: {json.dumps({'type': 'token', 'content': low_confidence_msg})}\n\n"
            
            # Signal end of stream
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/extract")
async def extract_course(request: ExtractRequest):
    """Extract structured course information (credits, exam pattern, etc.) as JSON."""
    try:
        result = extract_course_info(request.course_query, top_k=7)
        
        if not result["success"]:
            raise HTTPException(
                status_code=422,
                detail=result["error"] or "Extraction failed"
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during extraction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
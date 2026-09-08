from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .agent import agent
from .config import APP_NAME, APP_VERSION, FRONTEND_DIR
from .memory import memory
from .rag import rag

app = FastAPI(title=APP_NAME, version=APP_VERSION, description="Agentic AI Student Support Assistant using RAG, tools and conversational memory.")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=100)

class ChatResponse(BaseModel):
    answer: str
    intent: dict
    tool_result: Optional[dict] = None
    sources: list

@app.get("/api/health")
def health():
    return {"status": "online", "application": APP_NAME, "version": APP_VERSION, "documents": len(rag.documents)}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        return agent.execute(request.session_id, message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/memory/clear")
def clear_memory(session_id: str = "default"):
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/api/documents")
def documents():
    names = sorted({item["source"] for item in rag.documents})
    return {"documents": names, "count": len(names)}

@app.get("/")
def serve_frontend():
    return FileResponse(Path(FRONTEND_DIR) / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

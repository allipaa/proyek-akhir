from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .kb_generator import generate_kb_draft
from .kb_analyzer import (
    get_top_issues,
    get_category_stats,
    get_summary,
    get_top_detailed
)

app = FastAPI(
    title="Knowledge Base Otomatis CloudKilat",
    description="RAG dengan LangChain + Gemini + ChromaDB",
    version="2.0.0"
)

# CORS (agar bisa dipakai frontend / Gradio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODEL
# =========================
class GenerateRequest(BaseModel):
    topic: str


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "status": "running",
        "project": "KB Otomatis CloudKilat",
        "version": "2.0 (LangChain)"
    }


# =========================
# GENERATE KB (RAG + GEMINI)
# =========================
@app.post("/generate-kb")
def generate_kb(req: GenerateRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topik tidak boleh kosong")

    try:
        result = generate_kb_draft(req.topic)
        return {
            "success": True,
            "data": result
        }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# ANALYZER ENDPOINTS
# =========================

@app.get("/analyze/top-issues")
def top_issues(n: int = 10):
    try:
        data = get_top_issues(n)
        return {
            "success": True,
            "data": data.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/categories")
def categories():
    try:
        data = get_category_stats()
        return {
            "success": True,
            "data": data.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/summary")
def summary():
    try:
        return {
            "success": True,
            "data": get_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/top-detailed")
def top_detailed(n: int = 5):
    try:
        data = get_top_detailed(n)
        return {
            "success": True,
            "data": data.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}
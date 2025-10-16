# server.py
import os, json
from fastapi.responses import Response
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from rag import RAGPipeline
from report_pdf import build_pdf_report  # unchanged
from dotenv import load_dotenv
#from fastapi.responses import HTMLResponse
#from report_html import build_html_page


load_dotenv()  # picks up .env

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

PIPE = RAGPipeline(index_dir="data/index")

@app.post("/ask")  # Structured JSON (LLM-backed if OPENAI_API_KEY present)
async def ask(question: str = Form(...), top_k: int = Form(6)):
    data = PIPE.answer(question, top_k=top_k)
    return data

@app.post("/ask_concise")
async def ask_concise(question: str = Form(...), top_k: int = Form(6)):
    text = PIPE.answer_concise(question, top_k=top_k)
    return {"text": text}

@app.post("/ask_exact")
async def ask_exact(question: str = Form(...), top_k: int = Form(6)):
    text = PIPE.answer_exact(question, top_k=top_k)
    return {"text": text}

@app.post("/report")
async def report(question: str = Form(...), top_k: int = Form(6)):
    data = PIPE.answer(question, top_k=top_k)
    pdf_bytes = build_pdf_report(question, data)
    return Response(pdf_bytes, media_type="application/pdf")

#@app.post("/report_html", response_class=HTMLResponse)
#async def report_html(question: str = Form(...), top_k: int = Form(6)):
#    data = PIPE.answer(question, top_k=top_k)
#    html = build_html_page(question, data)
#    return HTMLResponse(content=html, status_code=200)



@app.post("/reindex")
async def reindex():
    # rescans local folder + web list, rebuilds TF-IDF/FAISS
    from ingest import scan_and_index
    scan_and_index()
    # refresh in-memory pipeline vectorizer/index
    from vectorstore_faiss import FAISSStore
    PIPE.index = FAISSStore("data/index")
    return {"ok": True, "message": "Re-index complete"}

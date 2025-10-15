# server.py
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from rag import RAGPipeline

app = FastAPI(title="RIA Assistant API")
PIPE = RAGPipeline()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
async def ask(question: str = Form(...), top_k: int = Form(5)):
    data = PIPE.answer(question, top_k=top_k)
    return JSONResponse(data)

@app.post("/ask_concise")
async def ask_concise(question: str = Form(...), top_k: int = Form(5)):
    text = PIPE.answer_concise(question, top_k=top_k)
    return PlainTextResponse(text)

# >>> ADD THIS <<<
@app.post("/ask_exact")
async def ask_exact(question: str = Form(...), top_k: int = Form(6)):
    text = PIPE.answer_exact(question, top_k=top_k)
    return PlainTextResponse(text)

@app.post("/report")
async def report(question: str = Form(...), top_k: int = Form(6)):
    pdf_bytes = PIPE.generate_pdf(question, top_k=top_k)
    return StreamingResponse(iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ria_report.pdf"},
    )

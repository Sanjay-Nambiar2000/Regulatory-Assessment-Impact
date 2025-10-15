# Lifeblood RIA Assistant — Local RAG Starter Kit

A **local, Azure-free** Retrieval-Augmented Generation (RAG) prototype for the *Regulatory Impact Assessment Assistant*.
It uses only local Python libraries, FAISS vector search, and (optionally) a **local LLM via Ollama** (e.g., `llama3.1:8b-instruct`).
You can also point it at an OpenAI-compatible endpoint by setting environment variables.

> ✅ Designed to be portable and later swappable into Azure AI Foundry/Cognitive Search.
> ✅ Deterministic orientation with explicit, grounded citations (doc + page).
> ✅ Generates a structured **RIA assessment report** and downloadable **PDF**.

---

## 1) Quick Start

### A. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B. (Optional) Set up a local LLM (recommended)
Install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.1:8b-instruct
```
The server should run on `http://localhost:11434`. The code will use Ollama automatically if available.

### C. Ingest the provided documents
Put your PDFs/DOCs into `data/raw/`, then run:
```bash
python ingest.py
```
This will parse, chunk, embed, and index documents into `data/index/`.

### D. Run the API
```bash
uvicorn server:app --reload --port 8000
```
Swagger UI: http://127.0.0.1:8000/docs

### E. (Optional) Use the simple UI
```bash
streamlit run ui_streamlit.py
```
- Ask a question (e.g., “What are donor screening requirements for ocular tissue?”)
- Click **Generate PDF** to download a structured assessment report.

---

## 2) Deterministic, Grounded Output

- Retrieval uses **BM25 + dense embeddings (FAISS)**; seeds are fixed where applicable.
- Answers are assembled with a strict template that includes:
  - **Compliance status** (Yes/No/Unclear) with rationale
  - **Quoted evidence** with **document name + page** for each claim
  - **Violations/risks** and **alternative suggestions**
  - **Summary proposal** and **Human oversight flag**

> The code never *hallucinates* citations: it only uses retrieved passages and shows their source.

---

## 3) Swapping to Azure later

This repo isolates retrieval and LLM calls. When you get Azure access:
- Replace `vectorstore_faiss.py` with an Azure Cognitive Search retriever or hybrid retriever.
- Replace `llm.py` with an Azure OpenAI client (GPT‑4o + embeddings).

---

## 4) Environment Variables

You can route LLM calls by environment variables:

- **Ollama (default if running):**
  - `OLLAMA_MODEL` (default: `llama3.1:8b-instruct`)
  - `OLLAMA_BASE`  (default: `http://localhost:11434`)

- **OpenAI-Compatible (optional):**
  - `OPENAI_API_KEY`
  - `OPENAI_BASE` (optional; default is the OpenAI public endpoint)
  - `OPENAI_MODEL` (e.g., `gpt-4o-mini`)

If neither is available, the app will **fall back to a templated, non‑LLM summariser**, which still returns
grounded citations (useful for fully offline demos).

---

## 5) Data versioning concept

Every ingested file gets a **hash** and timestamp in `data/index/chunks.jsonl`. Old versions may co-exist
so you can keep historical context. Re-ingesting the same filename with changed content yields a new
`version_id`, allowing future “compare versions” features.

---

## 6) Security & PII

- This prototype keeps data **local**.
- It logs only minimal telemetry and never sends your documents outside unless you configure a remote LLM.
- When moving to production, integrate **Azure Entra ID (SSO)**, **Key Vault**, **Private Link**, logging to **Azure Monitor**.

---

## 7) References embedded in sample data

This starter includes TGA guidance PDFs you provided. Typical compliance checks will cite sections from:
- **TGO 109** – Standards for biologicals (general & specific requirements).
- **TGO 108** – Donor screening requirements.
- **Australian cGMP Code** – Blood, tissues, HCT products.
- **TMF Guidance** – Technical Master Files preparation.
See `/data/raw/` for the documents.

---

## 8) Project Layout

```
ria_assistant/
├── data/
│   ├── raw/                 # place source PDFs/DOCXs here
│   └── index/               # auto-created FAISS + chunk store
├── llm.py                   # Ollama/OpenAI wrapper + offline fallback
├── rag.py                   # retrieval + prompting + answer assembly
├── vectorstore_faiss.py     # simple FAISS store helpers
├── ingest.py                # parse → chunk → embed → index
├── server.py                # FastAPI service (POST /ask, POST /report)
├── report_pdf.py            # Markdown/sections → PDF
├── ui_streamlit.py          # lightweight UI
├── utils_pdf.py             # PDF text extraction helpers
├── requirements.txt
└── README.md
```

---

## 9) Example query

> “A change proposes storing amnion products at –20°C for 18 months. Is this compliant, and what regs apply?”

The assistant will retrieve relevant clauses (e.g., TGO 109 Part 7 storage requirements) and answer with
citations like: *TGO 109, Part 7, Section 49 — Storage & transportation (p. X)*, then give a compliance decision,
violations if any, alternatives, and a summary. Finally, you can download the response as a **PDF**.

---

**Enjoy building!** This is meant as a solid baseline you can iterate on with your stakeholders.

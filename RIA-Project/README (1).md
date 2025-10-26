# Rhonda — Regulatory Impact Assessment (RIA) Assistant

Deterministic, citation-grounded RAG that turns SOP/regulation checks into a first-pass, auditable answer with exact sources, confidence, conflict flags, and one-click PDF export.

##  What it does

- **Deterministic outputs** — fixed TF-IDF index, stable re-ranking, `temperature=0`, strict JSON schema.
- **Hybrid retrieval** — TF-IDF cosine + precedence-aware stable sort (law → regulator → accreditation → internal → unknown).
- **Conflict detection** — numeric clashes surfaced with a precedence-based resolution hint.
- **Traceable citations & versions** — title/page/section + version/effective_date + optional PDF export.
- **Simple UI** — single `index.html`; calls `/api/health`, `/api/query`, `/api/pdf`.

---

##  Project structure

```
.
├─ regulatory_docs/           # Your PDFs/DOCX/TXT/MD (source of truth)(in zip file)
├─ data/index/                # Built index(during running the code): chunks.jsonl, tfidf_vectorizer.pkl, vectors.npy (not uploading the data     beacuse of large data)
├─ ingest.py                  # Deterministic ingestion & TF-IDF index build
├─ hybrid_ria_backend.py      # FastAPI app (retrieval, conflicts, LLM, PDF, UI route)
├─ index.html                 # Single-page UI
└─ requirements.txt
```

---

##  Quickstart

### 1 Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 Configure env

Create `.env` at repo root (the backend loads it automatically):

```properties
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
DOCS_DIRECTORY=./regulatory_docs
INDEX_DIR=./data/index
CHUNK_SIZE=420
CHUNK_OVERLAP=60
```

### 3 Add documents & build index

Put source files under `regulatory_docs/` then:

```bash
python ingest.py
```

This scans, chunks, extracts metadata (hierarchy/version/effective), and saves TF-IDF artifacts to `data/index/`.

### 4 Run the app

```bash
# either
python hybrid_ria_backend.py
# or
uvicorn hybrid_ria_backend:app --host 127.0.0.1 --port 8000 --reload
```

### 5 Open the UI

- UI route: `http://127.0.0.1:8000/ui` (root `/` redirects to `/ui`).
- The page calls the API and provides PDF export.

---

## API

### `GET /api/health`
Readiness and chunk count.

### `POST /api/query`
Form fields: 
- `question` (string, required)  
- `top_k` (int, default 6)  
- `link_previous` (bool, default false — UI usually sends false)

Runs retrieval → conflict detection → LLM (temp=0, JSON mode) → normalizes citations/versions/conflicts/traceability.

**Returns (shape):**
```json
{
  "answer": "...",
  "compliance_status": "Compliant|Non-Compliant|Unclear",
  "confidence_score": 0.0,
  "confidence_level": "LOW|MEDIUM|HIGH",
  "citations": [ { "doc_title":"...", "page":12, "section":"...", "version":"...", "effective_date":"...", "quote":"..." } ],
  "violations": [],
  "recommendations": [],
  "traceability": [ ... ],
  "versions": [ { "doc_title":"...", "hierarchy":"...", "version":"...", "effective_date":"..." } ],
  "conflicts": [ ... ],
  "escalate_to_human": false,
  "metadata": { "num_sources": 6, "top_k": 6, "retrieval_method": "TF-IDF cosine", "timestamp": "..." }
}
```

### `POST /api/pdf`
Accepts the previous `/api/query` JSON (optionally `include_advanced: true`) and returns a PDF with evidence, versions, and clashes.

---

## Architecture (implemented)

```
                ┌─────────────────────────────────────────────────────┐
                │                 Ingestion (offline)                 │
                │  ingest.py                                         │
 Documents ───► │  - Chunk & normalize                               │
(regulatory_)   │  - Parse version/effective_date & hierarchy        │
                │  - Build TF-IDF (1–2 grams)                        │
                │  - Persist artifacts/ (vectorizer, vectors, meta)  │
                └─────────────────────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────────────────────┐
                │                 Backend (online)                    │
                │  hybrid_ria_backend.py                             │
                │  - Retrieve (TF-IDF cosine)                        │
                │  - Governance-aware re-rank (stable)               │
                │  - Conflict detection (numeric)                    │
                │  - LLM (temp=0, JSON schema)                       │
                │  - PDF export                                      │
                └─────────────────────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────────────────────┐
                │                      UI                             │
                │  index.html                                         │
                │  - Chat input                                       │
                │  - Citations, conflicts, versions                   │
                │  - Confidence bar & verdict chip                    │
                │  - Export PDF                                       │
                └─────────────────────────────────────────────────────┘
```

---

## Configuration

Environment variables used by the app:

- `OPENAI_API_KEY` (required) — used by the OpenAI client.
- `MODEL_NAME` (default `gpt-4o-mini`) — chat model name.
- `DOCS_DIRECTORY` (default `./regulatory_docs`) — ingestion input.
- `INDEX_DIR` (default `./data/index`) — artifacts output.
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking controls.

**Document naming tips (optional but helpful):** include `v3`, `v2024-07-01`, or `2024-07-01` in filenames to populate version/effective_date metadata automatically.

---

## Quick testing checklist

- [ ] Run `python ingest.py` — confirm `data/index/` has `chunks.jsonl`, `tfidf_vectorizer.pkl`, `vectors.npy`.
- [ ] `GET /api/health` → `{ "ready": true, "chunks": N }`.
- [ ] Ask a few queries in the UI; repeat the same query to confirm determinism.
- [ ] Introduce two contradictory sources (same unit) and see a clash flagged in **Conflicts** (and in exported PDF).

---

## Security & Privacy

- This pilot runs locally. No auth is implemented. Keep documents/API keys private.
- The model only receives retrieved passages + minimal context, not entire documents.

---

## Troubleshooting

- **Empty answers or citations** → ensure `regulatory_docs/` has files and re-run `python ingest.py`.
- **Ranking looks odd** → hierarchy affects stable re-rank; confirm doc types are labelled/parsed correctly.
- **CORS/UI errors** → if hosting backend on a different port/host, update the fetch URLs in `index.html`.
- **Inconsistent answers** → confirm `temperature=0` and JSON mode; ensure you’re not linking unrelated previous context.

---

## Roadmap (optional)

- BM25 fusion alongside TF-IDF (score-normalized).
- Calibrated confidence mixing retrieval strength + LLM confidence.
- Policy packs (jurisdictional bundles, time-bounded comparisons).
- Redlines/diffs across document versions in the UI.

---

## Acknowledgements

Built with FastAPI, ReportLab, and OpenAI’s Chat Completions API. Designed for auditability and low-ops deployment.

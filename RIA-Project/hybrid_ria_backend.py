#!/usr/bin/env python3
"""
Rhonda Pilot — Hybrid RAG Backend (deterministic, cited, conflict-aware, actionable)

Endpoints:
- GET /ui          : serves index.html
- GET /            : redirects to /ui
- GET /api/health  : readiness probe
- POST /api/query  : run RIA analysis (no small talk; always analyze)
- POST /api/pdf    : generate a downloadable PDF (respects `include_advanced`)
- GET /favicon.ico : optional favicon (stub)

Requires:
- OPENAI_API_KEY in .env or environment
- Index built by ingest.py (data/index/{chunks.jsonl, tfidf_vectorizer.pkl, vectors.npy})
"""

import os
import re
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Form, Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from dotenv import load_dotenv

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# LLM client
from openai import OpenAI

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
VEC_PATH = INDEX_DIR / "vectors.npy"
VEC_PICKLE = INDEX_DIR / "tfidf_vectorizer.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
TOP_K_DEFAULT = 6

PRECEDENCE = {"law": 0, "regulator": 1, "accreditation": 2, "internal": 3, "unknown": 4}

# ------------------------------------------------------------------------------
# INDEX LOADER / RETRIEVAL
# ------------------------------------------------------------------------------
class RAGIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks: List[Dict[str, Any]] = []

    def load(self) -> bool:
        if not (VEC_PATH.exists() and VEC_PICKLE.exists() and CHUNKS_PATH.exists()):
            return False
        self.matrix = np.load(VEC_PATH)
        with open(VEC_PICKLE, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.chunks = [
            json.loads(x) for x in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        ]
        return True

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        # Deterministic cosine similarity over TF-IDF vectors
        qv = self.vectorizer.transform([query]).toarray()[0]
        denom = np.linalg.norm(self.matrix, axis=1) * (np.linalg.norm(qv) + 1e-9)
        sims = (self.matrix @ qv) / np.maximum(denom, 1e-9)

        # Over-retrieve, then stable sort
        idx = np.argsort(-sims)[:max(20, top_k * 3)]
        candidates = []
        for i in idx:
            c = dict(self.chunks[i])
            c["score"] = float(sims[i])
            candidates.append(c)

        candidates.sort(
            key=lambda x: (
                -round(x["score"], 6),
                PRECEDENCE.get(x.get("hierarchy", "unknown"), 9),
                (x.get("doc_title") or ""),
                (x.get("page") or 0),
                x.get("chunk_index") or 0,
            )
        )
        return candidates[:top_k]


RAG = RAGIndex()
RAG.load()

# ------------------------------------------------------------------------------
# LLM ANALYSIS (DETERMINISTIC)
# ------------------------------------------------------------------------------
rules = """You are Rhonda, a Regulatory Impact Assessment assistant for Australian Red Cross Lifeblood.
You must reason ONLY over the passages provided. Apply precedence strictly: law > regulator > accreditation > internal SOP.
When sources conflict, prefer the higher-precedence source and set "review_required": true if ambiguity remains.
OUTPUT: valid JSON with exactly these keys:
{
  "compliance_status": "Compliant" | "Non-Compliant" | "Unclear",
  "confidence_score": 0.0,
  "rationale": "short, structured explanation",
  "citations": [{
      "doc_title": "",
      "hierarchy": "law|regulator|accreditation|internal|unknown",
      "effective_date": null,
      "version": null,
      "section": null,
      "page": null,
      "quote": ""
  }],
  "violations_or_risks": [],
  "alternative_suggestions": [],
  "summary_proposal": "",
  "review_required": false
}
Cite ONLY from provided passages. Use exact quotes and include section/page if present.
ALTERNATIVES: Always propose practical, specific alternatives/mitigations that would move the request toward compliance,
anchored to the higher-precedence source. If conflicts exist, align the recommendation to the winning source.
If evidence is insufficient or contradictory at the same precedence level, set "compliance_status":"Unclear" and "review_required":true.
"""

# capture common numeric facts + units
_NUM_UNIT = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>months?|month|days?|day|hours?|hour|°c|degc|c|%|percent|ml|mL|µl|uL|L|ppm)\b",
    re.I,
)

def _unit_norm(u: str) -> str:
    u = (u or "").lower()
    return {
        "degc": "°c",
        "c": "°c",
        "percent": "%",
        "ml": "mL",
        "ul": "µL",
        "uL".lower(): "µL",
        "l": "L",
        "month": "months",
        "day": "days",
        "hour": "hours",
    }.get(u, u)

def extract_numeric_facts(text: str) -> List[Tuple[float, str]]:
    out = []
    for m in _NUM_UNIT.finditer(text or ""):
        try:
            v = float(m.group("val"))
        except Exception:
            continue
        unit = _unit_norm(m.group("unit"))
        out.append((v, unit))
    return out

def detect_conflicts(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If two passages mention numeric facts with the same unit but different values,
    surface a structured conflict with precedence-based resolution hint.
    """
    conflicts: List[Dict[str, Any]] = []
    for i in range(len(passages)):
        a = passages[i]
        a_nums = extract_numeric_facts(a.get("text", ""))
        if not a_nums:
            continue
        for j in range(i + 1, len(passages)):
            b = passages[j]
            b_nums = extract_numeric_facts(b.get("text", ""))
            if not b_nums:
                continue

            # compare by unit overlap
            units_a = {u for _, u in a_nums}
            units_b = {u for _, u in b_nums}
            overlap = units_a & units_b
            if not overlap:
                continue

            # For each overlapping unit, compare representative values (first seen)
            for unit in overlap:
                va = next((v for v, u in a_nums if u == unit), None)
                vb = next((v for v, u in b_nums if u == unit), None)
                if va is None or vb is None or abs(va - vb) < 1e-9:
                    continue

                # Build conflict record
                prec_a = PRECEDENCE.get(a.get("hierarchy", "unknown"), 99)
                prec_b = PRECEDENCE.get(b.get("hierarchy", "unknown"), 99)
                winner = "a" if prec_a < prec_b else "b" if prec_b < prec_a else "tie"

                what = f"{va:g} {unit} vs {vb:g} {unit}"
                topic_hint = (a.get("locator") or b.get("locator") or "")[:80] or "Requirement interval/value"

                resolution = None
                if winner == "a":
                    resolution = f"Prefer '{a.get('doc_title')}' ({a.get('hierarchy')}) over '{b.get('doc_title')}' ({b.get('hierarchy')}). Use {va:g} {unit}."
                elif winner == "b":
                    resolution = f"Prefer '{b.get('doc_title')}' ({b.get('hierarchy')}) over '{a.get('doc_title')}' ({a.get('hierarchy')}). Use {vb:g} {unit}."
                else:
                    resolution = "Same precedence; escalate for decision. Choose the stricter value or seek regulator guidance."

                conflicts.append({
                    "topic_hint": topic_hint,
                    "unit": unit,
                    "what_clashes": what,
                    "a": {
                        "doc": a.get("doc_title") or a.get("source"),
                        "hierarchy": a.get("hierarchy"),
                        "page": a.get("page"),
                        "section": a.get("locator"),
                        "value": va,
                        "unit": unit,
                        "quote": (a.get("text", "")[:220]).strip(),
                    },
                    "b": {
                        "doc": b.get("doc_title") or b.get("source"),
                        "hierarchy": b.get("hierarchy"),
                        "page": b.get("page"),
                        "section": b.get("locator"),
                        "value": vb,
                        "unit": unit,
                        "quote": (b.get("text", "")[:220]).strip(),
                    },
                    "resolution_hint": resolution,
                })
    return conflicts

def build_versions(passages: List[Dict[str, Any]]):
    out = {}
    for p in passages:
        k = p.get("doc_title") or p.get("source")
        if not k:
            continue
        out[k] = {
            "doc_title": k,
            "hierarchy": p.get("hierarchy", "unknown"),
            "version": p.get("version"),
            "effective_date": p.get("effective_date"),
        }
    return list(out.values())

def normalize_citations(passages, citations):
    if citations:
        return citations
    normed = []
    for p in passages[:3]:
        normed.append(
            {
                "doc_title": p.get("doc_title") or p.get("source"),
                "hierarchy": p.get("hierarchy", "unknown"),
                "effective_date": p.get("effective_date"),
                "version": p.get("version"),
                "section": p.get("locator") or None,
                "page": p.get("page"),
                "quote": (p.get("text", "")[:220]).strip(),
            }
        )
    return normed

def build_traceability(question: str, analysis: Dict[str, Any]):
    verdict = analysis.get("compliance_status", "Unclear")
    status = "satisfied" if verdict == "Compliant" else ("violated" if verdict == "Non-Compliant" else "unclear")
    cits = analysis.get("citations", [])
    rows = [
        {
            "requirement": question[:140],
            "status": status,
            "citations": [
                {
                    "doc_title": c.get("doc_title") or c.get("source"),
                    "section": c.get("section"),
                    "page": c.get("page"),
                }
                for c in cits
            ],
        }
    ]
    return rows

def generate_analysis(question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build conflict records first and pass summaries into the LLM for better alternatives
    conflicts = detect_conflicts(passages)
    conflict_summaries = [
        {
            "topic": c.get("topic_hint"),
            "what_clashes": c.get("what_clashes"),
            "winner": (
                (c["a"]["doc"], c["a"]["hierarchy"])
                if "Use" in (c.get("resolution_hint") or "")
                and c["a"]["doc"] in (c.get("resolution_hint") or "")
                else (c["b"]["doc"], c["b"]["hierarchy"])
                if c.get("resolution_hint", "").count("Prefer '") == 1 and c["b"]["doc"] in c["resolution_hint"]
                else None
            ),
            "resolution_hint": c.get("resolution_hint"),
        }
        for c in conflicts
    ]

    client = OpenAI()
    content = [
        {"role": "system", "content": rules},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request_text": question,
                    "precedence": ["law", "regulator", "accreditation", "internal"],
                    "conflicts_detected": conflict_summaries,
                    "passages": [
                        {
                            "doc_title": p.get("doc_title") or p.get("source"),
                            "hierarchy": p.get("hierarchy", "unknown"),
                            "effective_date": p.get("effective_date"),
                            "version": p.get("version"),
                            "page": p.get("page"),
                            "section": p.get("locator"),
                            "text": p.get("text"),
                        }
                        for p in passages
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=content,
    )
    try:
        analysis = json.loads(resp.choices[0].message.content)
    except Exception:
        analysis = {
            "compliance_status": "Unclear",
            "confidence_score": 0.5,
            "rationale": "Unable to parse model JSON; using retrieved evidence only.",
            "citations": [],
            "violations_or_risks": [],
            "alternative_suggestions": [],
            "summary_proposal": "See evidence.",
            "review_required": True,
        }

    # Normalize + enrich
    analysis["citations"] = normalize_citations(passages, analysis.get("citations"))
    score = float(analysis.get("confidence_score", 0.5))
    analysis["confidence_score"] = score
    analysis["confidence_level"] = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.6 else "LOW"
    analysis["versions"] = build_versions(passages)
    analysis["conflicts"] = conflicts  # detailed, structured
    analysis["traceability_matrix"] = build_traceability(question, analysis)

    # If AI didn't propose alternatives but a clear winner exists, nudge with a fallback suggestion
    if not analysis.get("alternative_suggestions"):
        for c in conflicts:
            hint = c.get("resolution_hint")
            if hint and "Prefer" in hint and "Use " in hint:
                analysis.setdefault("alternative_suggestions", []).append(
                    f"Align to {hint.split('Prefer ')[1].split(' over ')[0]} requirement: adopt {c['what_clashes'].split(' vs ')[-1]} if that reflects the winner, "
                    f"update SOPs accordingly, and document rationale referencing {c['a']['doc']} / {c['b']['doc']}."
                )

    return analysis

# ------------------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------------------
app = FastAPI(title="Rhonda Pilot — Hybrid RAG")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = ROOT / "favicon.ico"
    if path.exists():
        return FileResponse(str(path))
    return Response(content=b"", media_type="image/x-icon")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return (ROOT / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    ready = RAG.vectorizer is not None and RAG.matrix is not None and len(RAG.chunks) > 0
    return {"ready": ready, "chunks": len(RAG.chunks)}

@app.post("/api/query")
def process_query(question: str = Form(...), top_k: int = Form(TOP_K_DEFAULT)):
    try:
        if not RAG.vectorizer:
            raise HTTPException(status_code=400, detail="Index not built. Run document ingestion first.")
        passages = RAG.retrieve(question, top_k=top_k)
        if not passages:
            raise HTTPException(status_code=404, detail="No relevant passages found. Add more documents or refine the query.")
        analysis = generate_analysis(question, passages)
        return {
            "answer": analysis.get("rationale", ""),
            "compliance_status": analysis.get("compliance_status", "Unclear"),
            "confidence_score": analysis.get("confidence_score", 0.5),
            "confidence_level": analysis.get("confidence_level", "LOW"),
            "citations": analysis.get("citations", []),
            "violations": analysis.get("violations_or_risks", []),
            "recommendations": analysis.get("alternative_suggestions", []),
            "traceability": analysis.get("traceability_matrix", []),
            "versions": analysis.get("versions", []),
            "conflicts": analysis.get("conflicts", []),
            "escalate_to_human": analysis.get("review_required", True),
            "metadata": {
                "num_sources": len(passages),
                "top_k": top_k,
                "retrieval_method": "TF-IDF cosine",
                "timestamp": datetime.now().isoformat(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------
# PDF EXPORT (plain-language headings + optional advanced sections, with clash detail)
# ------------------------------------------------------------------------------
def _para(text: str, styles):
    return Paragraph((text or "").replace("\n", "<br/>"), styles["Normal"])

@app.post("/api/pdf")
def pdf_generate(analysis: Dict[str, Any] = Body(...)):
    """
    Generate a PDF report from the latest assistant analysis payload.
    Payload should match what /api/query returns. Front-end may set:
      analysis['include_advanced'] = True  -> include Evidence checklist, Versions, Clashes
    """
    include_advanced = bool(analysis.get("include_advanced", False))  # default OFF

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = Path(tempfile.gettempdir()) / f"RIA_Report_{ts}.pdf"

    styles = getSampleStyleSheet()
    heading = styles["Heading1"]
    subhead = styles["Heading2"]
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Regulatory Impact Assessment", heading))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Spacer(1, 14))

    status = analysis.get("compliance_status", "Unclear")
    conf = analysis.get("confidence_score", 0.0)
    story.append(Paragraph("Verdict", subhead))
    story.append(_para(f"<b>Status:</b> {status}", styles))
    story.append(_para(f"<b>Confidence:</b> {(conf*100):.0f}% ({analysis.get('confidence_level','')})", styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Why this verdict", subhead))
    story.append(_para(analysis.get("answer") or analysis.get("rationale", ""), styles))
    story.append(Spacer(1, 10))

    violations = analysis.get("violations") or analysis.get("violations_or_risks") or []
    if violations:
        story.append(Paragraph("Risks and non-compliances found", subhead))
        for v in violations:
            story.append(_para(f"• {v}", styles))
        story.append(Spacer(1, 8))

    recs = analysis.get("recommendations") or analysis.get("alternative_suggestions") or []
    if recs:
        story.append(Paragraph("Practical alternatives / mitigations", subhead))
        for r in recs:
            story.append(_para(f"• {r}", styles))
        story.append(Spacer(1, 8))

    cits = analysis.get("citations", [])
    if cits:
        story.append(Paragraph("Evidence (exact sources)", subhead))
        for i, c in enumerate(cits, 1):
            t = c.get("doc_title") or c.get("source") or "Unknown"
            sec = c.get("section") or ""
            pg  = f"(p.{c.get('page')})" if c.get("page") else ""
            ver = f" — {c.get('version')}" if c.get("version") else ""
            eff = f" — effective {c.get('effective_date')}" if c.get("effective_date") else ""
            story.append(_para(f"<b>[{i}] {t} {sec} {pg}{ver}{eff}</b>", styles))
            if c.get("quote"):
                story.append(_para(f"“{c['quote']}”", styles))
        story.append(Spacer(1, 10))

    # ----- Advanced (optional) -----
    if include_advanced:
        # Evidence checklist
        trace = analysis.get("traceability") or analysis.get("traceability_matrix") or []
        if trace:
            story.append(Paragraph("Evidence checklist (what we checked)", subhead))
            for row in trace:
                story.append(_para(f"<b>Requirement:</b> {row.get('requirement','')}", styles))
                story.append(_para(f"<b>Status:</b> {row.get('status','')}", styles))
                c = row.get("citations") or []
                for ct in c:
                    line = f"• {ct.get('doc_title','')}"
                    if ct.get('section'): line += f" {ct['section']}"
                    if ct.get('page'):    line += f" (p.{ct['page']})"
                    story.append(_para(line, styles))
                story.append(Spacer(1, 6))

        # Versions
        vers = analysis.get("versions", [])
        if vers:
            story.append(Paragraph("Source versions", subhead))
            for v in vers:
                line = f"{v.get('doc_title','')}"
                if v.get('hierarchy'): line += f" — {v['hierarchy']}"
                if v.get('version'):   line += f" — {v['version']}"
                if v.get('effective_date'): line += f" (effective {v['effective_date']})"
                story.append(_para(line, styles))
            story.append(Spacer(1, 8))

        # Possible source clashes (with detail)
        confs = analysis.get("conflicts", [])
        if confs:
            story.append(Paragraph("Possible source clashes", subhead))
            for c in confs:
                story.append(_para(f"<b>What clashes:</b> {c.get('what_clashes','')}", styles))
                story.append(_para(f"• A: {c['a'].get('doc','')} ({c['a'].get('hierarchy','')}) "
                                   f"{' '+(c['a'].get('section') or '')} "
                                   f"{'(p.'+str(c['a'].get('page'))+')' if c['a'].get('page') else ''}", styles))
                story.append(_para(f"• B: {c['b'].get('doc','')} ({c['b'].get('hierarchy','')}) "
                                   f"{' '+(c['b'].get('section') or '')} "
                                   f"{'(p.'+str(c['b'].get('page'))+')' if c['b'].get('page') else ''}", styles))
                if c.get("resolution_hint"):
                    story.append(_para(f"<i>Suggested resolution:</i> {c['resolution_hint']}", styles))
                story.append(Spacer(1, 6))

    # Build PDF
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="Regulatory Impact Assessment")
    doc.build(story)
    filename = f"RIA_Report_{ts}.pdf"
    return FileResponse(str(out_path), media_type="application/pdf", filename=filename)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hybrid_ria_backend:app", host="127.0.0.1", port=8000, reload=True)

# rag.py — TF-IDF + FAISS (CPU-only) with query expansion, citation normalisation, and safe backfills
import os, json, random, re
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from vectorstore_faiss import FAISSStore
from llm import generate

random.seed(42); np.random.seed(42)

# ----------------- utilities -----------------
def _split_into_chunks(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        if chunk: chunks.append(" ".join(chunk))
        i += max(1, max_tokens - overlap)
    return chunks

def build_bm25(corpus: List[str]) -> BM25Okapi:
    tokenized = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized)

def _short_name(path: str) -> str:
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]

# --- query expansion (lightweight, keyword based) ---
_TEMP_SYNONYMS = [
    "2–8 °C", "2 - 8 °C", "2 to 8 °C", "2-8 °C", "2-8C",
    "refrigerated", "cold chain", "controlled temperature",
    "storage and transportation", "transport", "shipping", "Section 24", "TGO 109"
]
_MALARIA_SYNONYMS = [
    "malaria", "malaria-endemic", "endemic region", "ineligib", "deferr", "period of ineligibility"
]
_DEFERRAL_SYNONYMS = [
    "deferral", "period of ineligibility", "ineligible", "donor eligibility", "exclusion", "contraindicated"
]

def _expand_query(q: str) -> str:
    ql = q.lower()
    extra = []
    if any(k in ql for k in ["2–8", "2-8", "2 to 8", "refriger", "cold chain", "transport", "shipping", "storage"]):
        extra += _TEMP_SYNONYMS
    if "malaria" in ql or "endemic" in ql:
        extra += _MALARIA_SYNONYMS
    if any(k in ql for k in ["deferr", "ineligib", "exclusion", "eligib"]):
        extra += _DEFERRAL_SYNONYMS
    if not extra:
        return q
    # join unique extras to avoid overly long query
    extra = list(dict.fromkeys(extra))[:12]
    return q + " " + " ".join(extra)

# ---------------- Embeddings wrapper ----------------
class TFIDFEmbedder:
    def __init__(self, path: str):
        self.path = path
        self.vec: TfidfVectorizer = None
        if os.path.exists(path):
            self.vec = joblib.load(path)

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.vec is None:
            raise RuntimeError("TFIDF vectorizer not loaded; run ingest.py first.")
        X = self.vec.transform(texts).astype(np.float32)
        return X.toarray()

# ---------------- RAG pipeline ----------------
class RAGPipeline:
    def __init__(self, index_dir: str = "data/index"):
        """
        index_dir must contain:
          - tfidf_vectorizer.pkl
          - index.faiss
          - faiss_meta.jsonl
        (created by ingest.py)
        """
        self.index_dir = index_dir
        self.index = FAISSStore(index_dir)
        self.embedder = TFIDFEmbedder(os.path.join(index_dir, "tfidf_vectorizer.pkl"))

    # ---------- helpers ----------
    def _short(self, s: str, n: int = 460) -> str:
        s = " ".join(s.split())
        return s if len(s) <= n else s[:n].rstrip(" ,;:.") + " …"

    def _cite_label(self, rec: dict) -> str:
        base = _short_name(rec["source"]).replace(".pdf", "")
        clause = rec.get("clause")
        if clause:
            return f"{base} ({clause}, p.{rec['page']})"
        return f"{base} (p.{rec['page']})"

    # ---------- core ----------
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.transform(texts)

    def retrieve(self, question: str, top_k: int = 6) -> List[Dict]:
        # require fitted vectorizer
        if self.embedder.vec is None:
            return []
        q_expanded = _expand_query(question)
        qvec = self.embed([q_expanded])  # (1, D)
        dense_hits = self.index.search(qvec, top_k=top_k*2)
        if not dense_hits:
            return []
        corpus = [h["text"] for h in dense_hits]
        bm25 = build_bm25(corpus)
        scores = bm25.get_scores(q_expanded.lower().split())
        order = np.argsort(-scores)[:top_k]
        return [dense_hits[i] | {"_bm25": float(scores[i])} for i in order]

    # ---------- exact / concise (deterministic, no LLM) ----------
    def answer_exact(self, question: str, top_k: int = 6) -> str:
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return "No matching clauses found. Refine the question or ingest more docs. [Human supervision required]"
        bullets, seen = [], set()
        for p in passages:
            txt = self._short(p["text"], 360)
            cite = self._cite_label(p)
            key = (txt, cite)
            if key in seen:
                continue
            seen.add(key)
            bullets.append(f"• {txt}  [{cite}]")
        return "Exact clauses & citations (review before use):\n\n" + "\n".join(bullets[:8]) + "\n\nHuman supervision required."

    def answer_concise(self, question: str, top_k: int = 5) -> str:
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return ("Unclear — no matching clauses found in the local index. "
                    "Refine the question or ingest more docs. [Human review required]")
        snippets, cites = [], []
        for p in passages[:4]:
            t = " ".join(p["text"].split())[:180].rstrip(" ,;:") + "."
            if t not in snippets:
                snippets.append(t)
            cites.append(self._cite_label(p))
        body = " ".join(snippets[:2])
        cite_str = "; ".join(dict.fromkeys(cites))
        first = " ".join(passages[0]["text"].lower().split())
        if any(k in first for k in ["must not", "not permitted", "ineligible", "contraindicated"]):
            verdict = "Not compliant (likely)"
        elif any(k in first for k in ["must ", "required", "shall "]):
            verdict = "Possibly compliant (depends on specifics)"
        else:
            verdict = "Unclear"
        return f"{verdict}: {body} [Citations: {cite_str}]. Human supervision required."

    # ---------- structured (LLM, JSON) ----------
    def format_prompt(self, question: str, passages: List[Dict]) -> str:
        rules = (
            "You are a Regulatory Impact Assessment assistant for Australian Lifeblood.\n"
            "Answer strictly using ONLY the quoted passages.\n"
            "Include citations as [Document (page X)]. If unclear, say so and require human supervision.\n"
            "Return a JSON object in the exact schema shown."
        )
        quotes = []
        for i, p in enumerate(passages, 1):
            quotes.append(f"[{i}] {p['source']} (page {p['page']})\n\"\"\"\n{p['text']}\n\"\"\"\n")
        return f"""{rules}

Question:
{question}

Evidence passages:
{chr(10).join(quotes)}

Your response MUST be a single JSON object matching this schema and constraints:
{{
  "compliance_status": "Compliant" | "Not Compliant" | "Unclear",
  "rationale": "1–3 sentences grounded ONLY in the quotes",
  "citations": [
    {{"source": "filename.pdf", "page": <int>, "quote": "verbatim clause"}}
  ],
  "violations_or_risks": [
    "At least 1 specific risk or violated clause; if unclear, state the most likely risk."
  ],
  "alternative_suggestions": [
    "At least 1 concrete mitigation (e.g., validated 2–8 °C shipper, temp logger)."
  ],
  "summary_proposal": "Short summary referencing cited clauses.",
  "human_supervision_required": true
}}

Rules:
- Do NOT invent citations; every assertion must be backed by the provided quotes.
- If the user asks to LIST requirements (e.g., “what must/include/requirements”), DO NOT assess current compliance.
  In that case: set "compliance_status" to "Compliant" to indicate the requirement extraction is complete,
  provide the requirements and citations, and set "human_supervision_required" to false.
- If evidence is insufficient, set "Unclear" BUT STILL provide at least one plausible risk and one mitigation suggestion.
- Keep quotes short (1–3 lines) and include the page numbers.
"""

    def _backfill_risks_suggestions(self, question: str) -> Dict[str, List[str]]:
        q = question.lower()
        risks, alts = [], []
        if any(k in q for k in ["2–8", "2-8", "2 to 8", "refriger", "cold chain", "transport", "shipping"]):
            risks += [
                "Risk of temperature excursion during transport leading to product quality compromise.",
                "Potential non-compliance with storage/transport clauses if 2–8 °C cannot be maintained for full transit.",
            ]
            alts += [
                "Use a validated 2–8 °C shipper with lane qualification and pre-conditioned gel packs.",
                "Include a calibrated temperature logger and define acceptance criteria; review on receipt.",
                "Reduce transit time or split shipments; ensure monitored handover at each leg.",
            ]
        if "malaria" in q or "endemic" in q:
            risks += [
                "Risk of transfusion-transmitted malaria if deferral/eligibility rules are not applied.",
            ]
            alts += [
                "Apply the specified deferral period or require negative testing per the cited appendix before acceptance.",
            ]
        if not risks:
            risks = ["Evidence insufficient; risk of non-compliance if policy deviates from cited clauses."]
        if not alts:
            alts = ["Escalate for human review and obtain written confirmation against the cited standard/appendix."]
        return {"risks": risks, "alts": alts}

    def answer(self, question: str, top_k: int = 6) -> Dict:
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            rf = self._backfill_risks_suggestions(question)
            return {
                "compliance_status": "Unclear",
                "rationale": "No index or no matches. Ingest documents first.",
                "citations": [],
                "violations_or_risks": rf["risks"],
                "alternative_suggestions": rf["alts"],
                "summary_proposal": "Ingest relevant documents and retry; see suggested mitigations.",
                "human_supervision_required": True
            }

        prompt = self.format_prompt(question, passages)
        text = generate(prompt)

        # Try to parse model JSON; fall back to safe default
        try:
            data = json.loads(text)
        except Exception:
            data = {
                "compliance_status": "Unclear",
                "rationale": text[:600],
                "citations": [
                    {"source": _short_name(p["source"]), "page": p["page"], "quote": " ".join(p["text"].split())[:300]}
                    for p in passages[:4]
                ],
                "violations_or_risks": [],
                "alternative_suggestions": [],
                "summary_proposal": "See grounded excerpts and consult a human reviewer.",
                "human_supervision_required": True
            }

        # ---- Normalise / backfill citations ----
        if not data.get("citations"):
            data["citations"] = [
                {"source": _short_name(p["source"]), "page": p["page"], "quote": " ".join(p["text"].split())[:300]}
                for p in passages[:3]
            ]
        else:
            # Ensure each citation has a real filename + page
            if passages:
                first_src = _short_name(passages[0]["source"])
                first_page = passages[0]["page"]
            else:
                first_src, first_page = "", 1
            for c in data["citations"]:
                src = c.get("source") or first_src
                if src in ("doc", "", None):
                    src = first_src
                c["source"] = _short_name(src)
                if "page" not in c or not c["page"]:
                    c["page"] = first_page
                # trim quote length
                if c.get("quote"):
                    c["quote"] = " ".join(str(c["quote"]).split())[:400]

        # ---- Backfill risks/suggestions when empty ----
        rf = self._backfill_risks_suggestions(question)
        if not data.get("violations_or_risks"):
            data["violations_or_risks"] = rf["risks"]
        if not data.get("alternative_suggestions"):
            data["alternative_suggestions"] = rf["alts"]

        # ---- Human supervision policy ----
        status = (data.get("compliance_status") or "Unclear").strip()
        data["human_supervision_required"] = (status != "Compliant")

        return data

# rag.py  — TF-IDF + FAISS (no torch)
import os, json, random, numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from vectorstore_faiss import FAISSStore
from llm import generate
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(42); np.random.seed(42)

VEC_PATH = "data/index/tfidf_vectorizer.pkl"

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

class TFIDFEmbedder:
    def __init__(self, path=VEC_PATH):
        self.path = path
        self.vec: TfidfVectorizer = None
        if os.path.exists(path):
            self.vec = joblib.load(path)

    def fit(self, texts: List[str]):
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                                   min_df=2, max_df=0.9, norm="l2")
        self.vec.fit(texts)

    def save(self):
        if self.vec:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            joblib.dump(self.vec, self.path)

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vec.transform(texts).astype(np.float32)
        return X.toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        self.save()
        X = self.vec.transform(texts).astype(np.float32)
        return X.toarray()

class RAGPipeline:
    def __init__(self, index_dir="data/index"):
        self.index = FAISSStore(index_dir)
        self.embedder = TFIDFEmbedder()

    # --- add inside class RAGPipeline -------------------------------------
    
    # --- concise mode helpers ---------------------------------
        # ---------- EXACT (clauses + citations) MODE ----------
    def _short(self, s: str, n: int = 240) -> str:
        s = " ".join(s.split())
        return s if len(s) <= n else s[:n].rstrip(" ,;:.") + " …"

    def _cite_label(self, src: str, page: int) -> str:
        base = src.rsplit("/", 1)[-1].replace(".pdf", "")
        return f"{base} p.{page}"

    def answer_exact(self, question: str, top_k: int = 6) -> str:
        """
        Deterministic: returns bullet-point clauses verbatim-ish with document+page citations.
        No LLM required.
        """
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return "No matching clauses found. Refine the question or ingest more docs. [Human supervision required]"

        bullets = []
        seen = set()
        for p in passages:
            txt = self._short(p["text"], 350)
            cite = self._cite_label(p["source"], p["page"])
            key = (txt, cite)
            if key in seen:
                continue
            seen.add(key)
            bullets.append(f"• {txt}  [{cite}]")

        header = "Exact clauses & citations (review before use):"
        footer = "Human supervision required."
        return header + "\n\n" + "\n".join(bullets[:8]) + "\n\n" + footer

    def _short_source(self, fn: str) -> str:
        base = fn.rsplit("/", 1)[-1].replace(".pdf", "")
        return (base[:28] + "…") if len(base) > 28 else base

    def answer_concise(self, question: str, top_k: int = 5) -> str:
        """Return a short, deterministic answer with inline citations (no LLM required)."""
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return ("Unclear — no matching clauses found in the local index. "
                    "Refine the question or ingest more docs. [Human review required]")

        # take up to 2 tight snippets from the best passages
        snippets, cites = [], []
        for p in passages[:4]:
            t = " ".join(p["text"].split())[:180].rstrip(" ,;:") + "."
            if t not in snippets:
                snippets.append(t)
            cites.append(f"{self._short_source(p['source'])} p.{p['page']}")
        body = " ".join(snippets[:2])
        cite_str = "; ".join(dict.fromkeys(cites))  # de-dup, keep order

        # very conservative verdict heuristic
        first = " ".join(passages[0]["text"].lower().split())
        if any(k in first for k in ["must not", "not permitted", "ineligible", "contraindicated"]):
            verdict = "Not compliant (likely)"
        elif any(k in first for k in ["must ", "required", "shall "]):
            verdict = "Possibly compliant (depends on specifics)"
        else:
            verdict = "Unclear"

        return f"{verdict}: {body} [Citations: {cite_str}]. Human supervision required."

    def embed(self, texts: List[str], fit: bool=False) -> np.ndarray:
        if fit or (self.embedder.vec is None):
            return self.embedder.fit_transform(texts)
        return self.embedder.transform(texts)

    def retrieve(self, question: str, top_k: int = 6) -> List[Dict]:
        if self.embedder.vec is None:
            return []
        qvec = self.embed([question])
        dense_hits = self.index.search(qvec, top_k=top_k*2)
        if not dense_hits: return []
        corpus = [h["text"] for h in dense_hits]
        bm25 = build_bm25(corpus)
        scores = bm25.get_scores(question.lower().split())
        order = np.argsort(-scores)[:top_k]
        return [dense_hits[i] | {"_bm25": float(scores[i])} for i in order]

    def format_prompt(self, question: str, passages: List[Dict]) -> str:
        rules = (
            "You are a Regulatory Impact Assessment assistant for Australian Lifeblood.\n"
            "Answer strictly using ONLY the quoted passages.\n"
            "Include citations as [Document (page X)]. If unclear, say so and require human supervision.\n"
            "Return a JSON object in the exact schema shown."
        )
        quotes = []
        for i,p in enumerate(passages,1):
            quotes.append(f"[{i}] {p['source']} (page {p['page']})\n\"\"\"\n{p['text']}\n\"\"\"\n")
        return f"""{rules}

Question:
{question}

Evidence passages:
{chr(10).join(quotes)}

Your response must be in this JSON schema:
{{
  "compliance_status": "Compliant|Not Compliant|Unclear",
  "rationale": "short reasoning grounded in the evidence",
  "citations": [{{"source": "doc", "page": 1, "quote": "..."}}],
  "violations_or_risks": ["..."],
  "alternative_suggestions": ["..."],
  "summary_proposal": "one paragraph summary",
  "human_supervision_required": true
}}
"""

    def answer(self, question: str, top_k: int = 6) -> Dict:
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return {
                "compliance_status":"Unclear",
                "rationale":"No index or no matches. Ingest documents first.",
                "citations":[], "violations_or_risks":[],
                "alternative_suggestions":[],
                "summary_proposal":"Ingest relevant documents and retry.",
                "human_supervision_required": True
            }
        prompt = self.format_prompt(question, passages)
        text = generate(prompt)
        try:
            data = json.loads(text)
        except Exception:
            data = {
                "compliance_status":"Unclear",
                "rationale": text[:600],
                "citations":[{"source":p["source"],"page":p["page"],"quote":p["text"][:400]} for p in passages[:4]],
                "violations_or_risks":[],
                "alternative_suggestions":[],
                "summary_proposal":"See grounded excerpts and consult a human reviewer.",
                "human_supervision_required": True
            }
        if not data.get("citations"):
            data["citations"] = [{"source":p["source"],"page":p["page"],"quote":p["text"][:400]} for p in passages[:4]]
            
        return data
        

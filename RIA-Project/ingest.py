#!/usr/bin/env python3
"""
Rhonda Pilot — Ingestion + Index Builder (deterministic)

- Scans ./regulatory_docs for PDF/DOCX/TXT/MD
- Extracts text by page/section, builds fixed-size chunks
- Adds rich metadata: hierarchy, version, effective_date, locator/page
- Builds deterministic TF-IDF vectors for hybrid retrieval
- Saves:
    data/index/chunks.jsonl
    data/index/tfidf_vectorizer.pkl
    data/index/vectors.npy
"""
import os, re, json, hashlib, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

# -------------------
# CONFIG (env or defaults)
# -------------------
ROOT = Path(__file__).resolve().parent
DOCS_DIRECTORY = Path(os.getenv("DOCS_DIRECTORY", ROOT / "regulatory_docs"))
OUT_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# -------------------
# HELPERS
# -------------------
def infer_hierarchy_from_name(name: str) -> str:
    n = name.lower()
    if " act" in n or n.endswith("act.pdf") or "regulation" in n:
        return "law"
    if "tga" in n or "therapeutic goods" in n or "gmp" in n or "guideline" in n or "guidance" in n:
        return "regulator"
    if "nata" in n or "accreditation" in n:
        return "accreditation"
    if "sop" in n or "policy" in n or "procedure" in n:
        return "internal"
    return "unknown"

def parse_version_and_date(name: str) -> Tuple[str, str]:
    # Try _vYYYY-MM-DD or -v3 / _v3; any YYYY-MM-DD as effective_date
    m = re.search(r"[_\-]v(\d{4}-\d{2}-\d{2})", name, re.I)
    version, effective_date = None, None
    if m:
        effective_date = m.group(1)
        version = f"v{effective_date}"
    else:
        m2 = re.search(r"[_\-]v(\d+)", name, re.I)
        if m2:
            version = f"v{m2.group(1)}"
    if not effective_date:
        m3 = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
        if m3:
            effective_date = m3.group(1)
    return version, effective_date

_LOCATOR_PATTERNS = [
    r"(Section|Clause|Part|Annex|Appendix)\s+([0-9A-Za-z\.\-\(\)]+)",
    r"(^|\b)(\d+(?:\.\d+)+[a-z]?)"
]

def extract_locator_hint(text: str) -> str:
    for pat in _LOCATOR_PATTERNS:
        m = re.search(pat, text or "", flags=re.I | re.M)
        if m:
            if m.lastindex and m.lastindex >= 2:
                return f"{m.group(1).title()} {m.group(2)}"
            return m.group(0).strip()
    return ""

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

# -------------------
# LOADERS
# -------------------
def load_pdf(path: Path) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    r = PdfReader(str(path))
    pages = []
    for i, p in enumerate(r.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"page": i+1, "text": txt})
    return pages

def load_docx(path: Path) -> List[Dict[str, Any]]:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(str(path))
    text = "\n".join([p.text for p in d.paragraphs])
    return [{"page": 1, "text": text}]

def load_txt(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"page": 1, "text": text}]

def chunk_text(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    toks = s.split()
    chunks = []
    i = 0
    while i < len(toks):
        chunk = toks[i:i+size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return chunks

# -------------------
# MAIN
# -------------------
def scan_docs() -> List[Dict[str, Any]]:
    docs = []
    for path in sorted(DOCS_DIRECTORY.glob("**/*")):
        if path.is_dir():
            continue
        name = path.name
        lower = name.lower()
        if lower.endswith(".pdf"):
            pages = load_pdf(path); ftype = "pdf"
        elif lower.endswith(".docx"):
            pages = load_docx(path); ftype = "docx"
        elif lower.endswith((".txt", ".md")):
            pages = load_txt(path); ftype = "text"
        else:
            continue

        version, effective = parse_version_and_date(name)
        hierarchy = infer_hierarchy_from_name(name)

        for pg in pages:
            docs.append({
                "file_name": name,
                "file_type": ftype,
                "file_path": str(path),
                "page": pg["page"],
                "text": pg["text"],
                "version": version,
                "effective_date": effective,
                "hierarchy": hierarchy
            })
    return docs

def build_index(docs: List[Dict[str, Any]]):
    all_chunks = []
    for d in docs:
        text = d.get("text") or ""
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for i, chunk_text_val in enumerate(chunks):
            chunk_id = sha256(f'{d["file_name"]}:{d["page"]}:{i}:{chunk_text_val[:40]}')
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_val,
                "source": d["file_name"],
                "doc_id": d["file_name"],
                "page": d["page"],
                "file_type": d["file_type"],
                "file_path": d["file_path"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_title": os.path.splitext(d["file_name"])[0],
                "hierarchy": d["hierarchy"],
                "version": d["version"],
                "effective_date": d["effective_date"],
                "locator": extract_locator_hint(chunk_text_val)
            })

    if not all_chunks:
        raise RuntimeError("No chunks produced. Are there readable docs in ./regulatory_docs ?")

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        max_features=50000
    )
    X = vectorizer.fit_transform(texts)

    (OUT_DIR / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks),
        encoding="utf-8"
    )
    with open(OUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    np.save(OUT_DIR / "vectors.npy", X.astype(np.float32).toarray())

    print(f"✅ Indexed docs: {len(set(c['doc_id'] for c in all_chunks))}")
    print(f"✅ Chunks: {len(all_chunks)}")
    print(f"✅ Vocab size: {len(vectorizer.vocabulary_)}")
    print(f"➡ Saved to {OUT_DIR}")

if __name__ == "__main__":
    print(f"Scanning {DOCS_DIRECTORY.resolve()}")
    docs = scan_docs()
    print(f"Loaded {len(docs)} doc-pages")
    build_index(docs)

# utils_ingest.py
import os, re, hashlib, datetime, requests
from typing import Iterable, List, Dict, Tuple, Optional

from pypdf import PdfReader
from docx import Document as DocxDocument  # pip install python-docx
from bs4 import BeautifulSoup              # pip install beautifulsoup4 lxml

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def sha256_text(t: str) -> str:
    return sha256_bytes(t.encode("utf-8", errors="ignore"))

def now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def read_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        out.append((i, txt.strip()))
    return out

def read_docx(path: str) -> List[Tuple[int, str]]:
    doc = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    # emulate "pages" as 1 block
    return [(1, text.strip())]

def read_txt(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    return [(1, t.strip())]

def read_web(url: str, timeout: int = 30) -> List[Tuple[int, str]]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # strip nav/script/style
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text("\n").split())
    return [(1, text.strip())]

def split_into_chunks(text: str, max_tokens: int = 420, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        if chunk:
            chunks.append(" ".join(chunk))
        i += max(1, max_tokens - overlap)
    return chunks

_CLAUSE_PAT = re.compile(
    r"(Section|Clause|Subclause|Subsection)\s+\d+(\.\d+)*", re.IGNORECASE
)

def guess_clause_label(s: str) -> Optional[str]:
    m = _CLAUSE_PAT.search(s)
    if m: 
        return m.group(0)
    return None

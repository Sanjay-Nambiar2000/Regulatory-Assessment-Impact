from pypdf import PdfReader
from typing import List, Dict

def extract_pdf_text(path: str) -> List[Dict]:
    """
    Returns list of dicts: [{"page": i+1, "text": "..."}]
    """
    out = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append({"page": i + 1, "text": txt})
    return out

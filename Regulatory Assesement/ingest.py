#!/usr/bin/env python3
"""
Rhonda Document Ingestion Script

Scans regulatory documents, extracts text, builds chunks, and creates TF-IDF index.
Supports: PDF, DOCX, TXT, MD files

Usage:
    python ingest.py
    
Output:
    data/index/chunks.jsonl
    data/index/tfidf_vectorizer.pkl
    data/index/vectors.npy
"""

import os
import re
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Document parsers
try:
    from pypdf import PdfReader
except ImportError:
    print("⚠️  pypdf not installed. PDF support disabled.")
    print("   Install with: pip install pypdf")
    PdfReader = None

try:
    import docx
except ImportError:
    print("⚠️  python-docx not installed. DOCX support disabled.")
    print("   Install with: pip install python-docx")
    docx = None

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DOCS_DIRECTORY = Path(os.getenv("DOCS_DIRECTORY", ROOT / "regulatory_docs"))
OUT_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# ------------------------------------------------------------------------------
# METADATA EXTRACTION
# ------------------------------------------------------------------------------
def infer_hierarchy_from_name(name: str) -> str:
    """Determine document hierarchy from filename"""
    n = name.lower()
    if " act" in n or n.endswith("act.pdf") or "regulation" in n or "legislation" in n:
        return "law"
    if "tga" in n or "therapeutic goods" in n or "gmp" in n or "guideline" in n or "guidance" in n:
        return "regulator"
    if "nata" in n or "accreditation" in n or "iso" in n:
        return "accreditation"
    if "sop" in n or "policy" in n or "procedure" in n or "internal" in n:
        return "internal"
    return "unknown"

def parse_version_and_date(name: str) -> Tuple[str, str]:
    """Extract version and effective date from filename"""
    version, effective_date = None, None
    
    # Try _vYYYY-MM-DD format
    m = re.search(r"[_\-]v(\d{4}-\d{2}-\d{2})", name, re.I)
    if m:
        effective_date = m.group(1)
        version = f"v{effective_date}"
    else:
        # Try _v3 or -v3 format
        m2 = re.search(r"[_\-]v(\d+)", name, re.I)
        if m2:
            version = f"v{m2.group(1)}"
    
    # Look for any date in filename
    if not effective_date:
        m3 = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
        if m3:
            effective_date = m3.group(1)
    
    return version, effective_date

def extract_locator_hint(text: str) -> str:
    """Extract section/clause reference from text"""
    patterns = [
        r"(Section|Clause|Part|Annex|Appendix|Article)\s+([0-9A-Za-z\.\-\(\)]+)",
        r"(^|\b)(\d+(?:\.\d+)+[a-z]?)"
    ]
    
    for pat in patterns:
        m = re.search(pat, text or "", flags=re.I | re.M)
        if m:
            if m.lastindex and m.lastindex >= 2:
                return f"{m.group(1).title()} {m.group(2)}"
            return m.group(0).strip()
    return ""

def sha256(s: str) -> str:
    """Generate SHA256 hash"""
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

# ------------------------------------------------------------------------------
# DOCUMENT LOADERS
# ------------------------------------------------------------------------------
def load_pdf(path: Path) -> List[Dict[str, Any]]:
    """Load PDF document"""
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Install with: pip install pypdf")
    
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"⚠️  Error extracting page {i+1}: {e}")
                text = ""
            pages.append({"page": i+1, "text": text})
        return pages
    except Exception as e:
        print(f"❌ Error loading PDF {path.name}: {e}")
        return []

def load_docx(path: Path) -> List[Dict[str, Any]]:
    """Load DOCX document"""
    if docx is None:
        raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
    
    try:
        doc = docx.Document(str(path))
        text = "\n".join([p.text for p in doc.paragraphs])
        return [{"page": 1, "text": text}]
    except Exception as e:
        print(f"❌ Error loading DOCX {path.name}: {e}")
        return []

def load_txt(path: Path) -> List[Dict[str, Any]]:
    """Load text document"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"page": 1, "text": text}]
    except Exception as e:
        print(f"❌ Error loading TXT {path.name}: {e}")
        return []

# ------------------------------------------------------------------------------
# TEXT CHUNKING
# ------------------------------------------------------------------------------
def chunk_text(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    tokens = s.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return chunks

# ------------------------------------------------------------------------------
# MAIN PROCESSING
# ------------------------------------------------------------------------------
def scan_docs() -> List[Dict[str, Any]]:
    """Scan document directory and extract content"""
    if not DOCS_DIRECTORY.exists():
        print(f"❌ Document directory not found: {DOCS_DIRECTORY}")
        print(f"   Create it and add regulatory documents:")
        print(f"   mkdir -p {DOCS_DIRECTORY}")
        return []
    
    docs = []
    supported_extensions = ['.pdf', '.docx', '.txt', '.md']
    files_found = list(DOCS_DIRECTORY.glob("**/*"))
    doc_files = [f for f in files_found if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not doc_files:
        print(f"❌ No documents found in {DOCS_DIRECTORY}")
        print(f"   Supported formats: PDF, DOCX, TXT, MD")
        return []
    
    print(f"📄 Found {len(doc_files)} documents")
    
    for path in sorted(doc_files):
        name = path.name
        lower = name.lower()
        
        print(f"  Processing: {name}...", end=" ")
        
        try:
            if lower.endswith(".pdf"):
                pages = load_pdf(path)
                ftype = "pdf"
            elif lower.endswith(".docx"):
                pages = load_docx(path)
                ftype = "docx"
            elif lower.endswith((".txt", ".md")):
                pages = load_txt(path)
                ftype = "text"
            else:
                print("⏭️  Skipped (unsupported)")
                continue
            
            if not pages:
                print("⚠️  No content extracted")
                continue
            
            version, effective = parse_version_and_date(name)
            hierarchy = infer_hierarchy_from_name(name)
            
            for pg in pages:
                docs.append({
                    "file_name": name,
                    "file_type": ftype,
                    "file_path": str(path.absolute()),
                    "page": pg["page"],
                    "text": pg["text"],
                    "version": version,
                    "effective_date": effective,
                    "hierarchy": hierarchy
                })
            
            print(f"✅ {len(pages)} pages")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return docs

def build_index(docs: List[Dict[str, Any]]):
    """Build search index from documents"""
    print("\n🔨 Building index...")
    
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
        raise RuntimeError("No chunks produced. Check if documents contain readable text.")
    
    print(f"  Created {len(all_chunks)} chunks")
    
    # Build TF-IDF vectors
    print("  Building TF-IDF vectors...")
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000
    )
    X = vectorizer.fit_transform(texts)
    
    # Save outputs
    print("  Saving index files...")
    (OUT_DIR / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks),
        encoding="utf-8"
    )
    
    with open(OUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    np.save(OUT_DIR / "vectors.npy", X.astype(np.float32).toarray())
    
    print("\n" + "="*70)
    print("✅ INDEX BUILD COMPLETE")
    print("="*70)
    print(f"Documents indexed: {len(set(c['doc_id'] for c in all_chunks))}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print("="*70)
    print("\n✅ You can now start the backend:")
    print("   python rhonda_complete.py")
    print()

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("📚 RHONDA DOCUMENT INGESTION")
    print("="*70)
    print(f"Document directory: {DOCS_DIRECTORY.absolute()}")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
    print("="*70)
    print()
    
    try:
        docs = scan_docs()
        if not docs:
            print("\n❌ No documents to process. Exiting.")
            print("\nTo get started:")
            print(f"  1. Create directory: mkdir -p {DOCS_DIRECTORY}")
            print(f"  2. Add PDF/DOCX documents to {DOCS_DIRECTORY}")
            print(f"  3. Run this script again: python ingest.py")
            exit(1)
        
        print(f"\n✅ Loaded {len(docs)} document pages")
        build_index(docs)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
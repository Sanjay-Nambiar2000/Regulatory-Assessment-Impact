# ingest.py
import os, json, glob, hashlib, time
from typing import List, Dict
from utils_pdf import extract_pdf_text
from rag import RAGPipeline, _split_into_chunks

RAW_DIR = "data/raw"
INDEX_DIR = "data/index"
META_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
os.makedirs(RAW_DIR, exist_ok=True); os.makedirs(INDEX_DIR, exist_ok=True)

def hash_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()[:16]

def main():
    pipe = RAGPipeline(index_dir=INDEX_DIR)

    # 1) collect ALL chunks first (so TF-IDF vocabulary sees the whole corpus)
    all_chunks, all_metas = [], []
    for path in glob.glob(os.path.join(RAW_DIR, "*")):
        if os.path.isdir(path) or not path.lower().endswith(".pdf"): 
            continue
        print("Scan:", path)
        version = hash_file(path)
        for pg in extract_pdf_text(path):
            for ch in _split_into_chunks(pg["text"], max_tokens=380, overlap=60):
                if not ch.strip(): continue
                all_chunks.append(ch)
                all_metas.append({
                    "source": os.path.basename(path),
                    "page": pg["page"],
                    "text": ch,
                    "version_id": version,
                    "ingested_at": int(time.time())
                })

    if not all_chunks:
        print("No PDFs found in data/raw")
        return

    # 2) fit TF-IDF once, transform all chunks, add to FAISS, persist metas
    print(f"Embedding {len(all_chunks)} chunks …")
    embs = pipe.embed(all_chunks, fit=True)
    pipe.index.add(embs, all_metas)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in all_metas:
            f.write(json.dumps(m, ensure_ascii=False)+"\n")
    print(f"Done. Chunks written: {len(all_chunks)}")

if __name__ == "__main__":
    main()

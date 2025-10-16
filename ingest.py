# ingest.py — multi-format ingest + versioning + TF-IDF index
import os, json, glob
from typing import Dict, List
from utils_ingest import (
    read_pdf, read_docx, read_txt, read_web, split_into_chunks,
    sha256_text, now_iso, guess_clause_label
)
from rag import RAGPipeline

RAW_DIR = "data/raw"
WEB_LIST = "data/web_urls.txt"  # optional: one URL per line
INDEX_DIR = "data/index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
META_PATH   = os.path.join(INDEX_DIR, "meta.jsonl")  # version manifest

os.makedirs(INDEX_DIR, exist_ok=True)

def _read_local_file(path: str):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        return read_pdf(path)           # List[(page, text)]
    if ext in [".docx"]:
        return read_docx(path)
    if ext in [".txt", ".md"]:
        return read_txt(path)
    return []

def _iter_sources() -> List[Dict]:
    items = []
    # local files
    for p in sorted(glob.glob(os.path.join(RAW_DIR, "*"))):
        base = os.path.basename(p)
        items.append({"type": "file", "id": base, "path": p})
    # urls
    if os.path.exists(WEB_LIST):
        with open(WEB_LIST, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                u = line.strip()
                if u:
                    items.append({"type": "url", "id": u, "url": u})
    return items

def _load_manifest() -> Dict[str, Dict]:
    if not os.path.exists(META_PATH):
        return {}
    d: Dict[str, Dict] = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                # skip bad/legacy line
                continue

            # Try to recover a doc_id from legacy shapes
            doc_id = (
                rec.get("doc_id")
                or rec.get("source")
                or rec.get("id")
                or rec.get("path")
                or rec.get("url")
            )
            if not doc_id:
                # unknown legacy row; skip
                continue

            vhash = rec.get("version_hash") or rec.get("hash") or rec.get("sha256")
            if not vhash:
                # if we truly can't find a version hash, skip
                continue

            # normalize to the new manifest shape
            norm = {
                "doc_id": doc_id,
                "version_hash": vhash,
                "created_at": rec.get("created_at") or rec.get("timestamp") or now_iso(),
                "source_kind": rec.get("source_kind") or ("url" if "url" in rec else "file"),
                "path": rec.get("path", ""),
                "url": rec.get("url", ""),
            }
            d.setdefault(doc_id, {})[vhash] = norm
    return d


def _append_manifest(rec: Dict):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _append_chunk(rec: Dict):
    with open(CHUNKS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def scan_and_index():
    manifest = _load_manifest()
    new_chunks: List[str] = []

    # reset chunks file (we’ll re-add only current versions)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    sources = _iter_sources()
    added_versions = 0
    total_chunks   = 0

    for s in sources:
        if s["type"] == "file":
            pages = _read_local_file(s["path"])
            source_name = os.path.basename(s["path"])
            full_text = "\n\n".join(t for _, t in pages)
            vhash = sha256_text(full_text)
            old = manifest.get(source_name, {})
            is_new = vhash not in old

            # write manifest entry (append-only; keeps history)
            rec_meta = {
                "doc_id": source_name,
                "version_hash": vhash,
                "created_at": now_iso(),
                "source_kind": "file",
                "path": s["path"],
            }
            if is_new:
                _append_manifest(rec_meta)
                added_versions += 1

            # (Re)chunk this version for the active index (we always keep latest active)
            # Choose the newest by created_at from existing + this rec.
            versions = list(old.values()) + ([rec_meta] if is_new else [])
            latest = sorted(versions, key=lambda r: r["created_at"], reverse=True)[0]
            if latest["version_hash"] != vhash:
                # current file isn't the latest; we still re-index the latest version from manifest
                # read the latest source again (simple path-based for pilot)
                pages = _read_local_file(latest["path"])

            # write chunks
            for page_num, page_txt in pages:
                for i, chunk in enumerate(split_into_chunks(page_txt)):
                    total_chunks += 1
                    _append_chunk({
                        "doc_id": source_name,
                        "version_hash": latest["version_hash"],
                        "page": page_num,
                        "chunk_id": f"{page_num}-{i}",
                        "text": chunk,
                        "source": source_name,
                        "clause": guess_clause_label(chunk),
                    })

        elif s["type"] == "url":
            pages = read_web(s["url"])
            full_text = "\n\n".join(t for _, t in pages)
            vhash = sha256_text(full_text)
            old = manifest.get(s["id"], {})
            is_new = vhash not in old
            rec_meta = {
                "doc_id": s["id"],
                "version_hash": vhash,
                "created_at": now_iso(),
                "source_kind": "url",
                "url": s["url"],
            }
            if is_new:
                _append_manifest(rec_meta)
                added_versions += 1

            versions = list(old.values()) + ([rec_meta] if is_new else [])
            latest = sorted(versions, key=lambda r: r["created_at"], reverse=True)[0]
            if latest["version_hash"] != vhash:
                pages = read_web(latest["url"])

            for page_num, page_txt in pages:
                for i, chunk in enumerate(split_into_chunks(page_txt)):
                    total_chunks += 1
                    _append_chunk({
                        "doc_id": s["id"],
                        "version_hash": latest["version_hash"],
                        "page": page_num,
                        "chunk_id": f"{page_num}-{i}",
                        "text": chunk,
                        "source": s["id"],
                        "clause": guess_clause_label(chunk),
                    })

    print(f"Added new versions: {added_versions}. Writing vector index...")

    # Build / update vector index
    from vectorstore_faiss import FAISSStore
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib, numpy as np

    texts, meta = [], []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"]); meta.append(r)

    if not texts:
        print("No chunks found. Add documents first.")
        return

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.9, norm="l2")
    X = vec.fit_transform(texts).astype("float32").toarray()

    os.makedirs(INDEX_DIR, exist_ok=True)
    joblib.dump(vec, os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl"))

    store = FAISSStore(INDEX_DIR)
    store.build(X, meta)
    print(f"Done. Chunks written: {len(texts)}")

if __name__ == "__main__":
    scan_and_index()

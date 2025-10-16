# vectorstore_faiss.py
import os, json
from typing import List, Dict, Optional
import numpy as np

try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu") from e


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    # normalize rows to unit length (avoid div-by-zero)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class FAISSStore:
    """
    Simple FAISS wrapper (IndexFlatIP + JSONL metadata).
    - build(vectors, meta): create a fresh index and save aligned metadata.
    - search(qvec, top_k): query by inner product (cosine when inputs are L2-normalized).
    """
    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.meta_path  = os.path.join(self.index_dir, "faiss_meta.jsonl")

        self.index: Optional[faiss.Index] = None
        self._meta_cache: Optional[List[Dict]] = None

    # ---------- persistence ----------
    def _save_meta(self, meta: List[Dict]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for rec in meta:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_meta(self) -> List[Dict]:
        if not os.path.exists(self.meta_path):
            return []
        out: List[Dict] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # skip bad rows
                    continue
        return out

    def _save_index(self, index: faiss.Index) -> None:
        faiss.write_index(index, self.index_path)

    def _load_index(self) -> Optional[faiss.Index]:
        if not os.path.exists(self.index_path):
            return None
        return faiss.read_index(self.index_path)

    def _ensure_loaded(self) -> None:
        if self.index is None:
            self.index = self._load_index()
        if self._meta_cache is None:
            self._meta_cache = self._load_meta()

    # ---------- public API ----------
    def build(self, vectors: np.ndarray, meta: List[Dict]) -> None:
        """
        Build a fresh index from dense vectors and aligned metadata.
        vectors: shape (N, D) float32
        meta:    list of length N with dicts, each must include at least:
                 {"text": ..., "source": ..., "page": int, ...}
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype="float32")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32", copy=False)

        if len(meta) != vectors.shape[0]:
            raise ValueError(f"meta length {len(meta)} does not match vectors {vectors.shape[0]}")

        # cosine similarity via inner product on L2-normalized vectors
        vecs = _l2_normalize(vectors)
        d = vecs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vecs)

        # persist
        self._save_index(index)
        self._save_meta(meta)

        # refresh in-memory
        self.index = index
        self._meta_cache = meta

    def search(self, qvec: np.ndarray, top_k: int = 6) -> List[Dict]:
        """
        qvec: shape (1, D) or (B, D) float32
        returns: list of top_k dicts for the first query row
        """
        self._ensure_loaded()
        if self.index is None or not self._meta_cache:
            return []

        if not isinstance(qvec, np.ndarray):
            qvec = np.asarray(qvec, dtype="float32")
        if qvec.dtype != np.float32:
            qvec = qvec.astype("float32", copy=False)
        if qvec.ndim == 1:
            qvec = qvec[None, :]

        q = _l2_normalize(qvec)
        k = min(top_k, len(self._meta_cache))
        scores, idxs = self.index.search(q, k)  # (B, k)
        idxs = idxs[0]; scores = scores[0]

        out: List[Dict] = []
        for i, sc in zip(idxs.tolist(), scores.tolist()):
            if i < 0:
                continue
            meta = self._meta_cache[i].copy()
            meta["score"] = float(sc)
            out.append(meta)
        return out

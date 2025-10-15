import faiss, os, json, numpy as np
from typing import List, Dict, Tuple

class FAISSStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")
        self.index = None
        self.dim = None

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if self.index is None:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.dim is None:
            self.dim = embeddings.shape[1]
        self.index.add(embeddings.astype(np.float32))
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for m in metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self._save()

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self._load()
        if self.index is None:
            return []
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        # read metas
        metas = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                metas = [json.loads(x) for x in f]
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metas):
                continue
            m = metas[idx]
            m["_score"] = float(dist)
            results.append(m)
        return results

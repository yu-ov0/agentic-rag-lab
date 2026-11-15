"""Dense retrieval with the BGE-small-zh ONNX encoder.

The corpus matrix is cached under data/index/ so repeated evaluation runs do
not re-encode 833 chunks. The corpus is tiny, so a normalized-matrix dot
product is faster than pulling in FAISS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.corpus import INDEX_DIR, ROOT, load_chunks

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunks: list[dict] | None = None,
        use_instruction: bool = False,
        model_path: Path | None = None,
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.chunks = chunks or load_chunks()
        self.use_instruction = use_instruction
        local = model_path or LOCAL_MODEL_DIR
        kwargs = {"specific_model_path": str(local)} if local.is_dir() else {}
        self.model = TextEmbedding(model_name=model_name, **kwargs)
        self.matrix = self._load_or_build_matrix()

    def _cache_path(self) -> Path:
        slug = self.model_name.replace("/", "__")
        return INDEX_DIR / f"dense_{slug}.npy"

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts, batch_size=32))
        mat = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-12, None)

    def _load_or_build_matrix(self) -> np.ndarray:
        cache = self._cache_path()
        if cache.exists():
            mat = np.load(cache)
            if mat.shape[0] == len(self.chunks):
                return mat
        mat = self._embed([c["text"] for c in self.chunks])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, mat)
        return mat

    def embed_query(self, query: str) -> np.ndarray:
        text = QUERY_INSTRUCTION + query if self.use_instruction else query
        vec = np.asarray(next(iter(self.model.embed([text]))), dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.matrix @ self.embed_query(query)
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results

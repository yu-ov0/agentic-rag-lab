"""Cross-encoder reranking on top of a first-stage retriever.

The rerank score is also the sufficiency signal: it is a direct (query, chunk)
relevance estimate, so the agent can ask "is the best chunk actually good
enough to answer this sub-question?" instead of trusting a cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

from rag.corpus import ROOT

LOCAL_MODEL_DIR = ROOT / "models" / "bge-reranker-v2-m3-int8"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3-int8"


class Reranker:
    def __init__(
        self,
        base,
        model_name: str = DEFAULT_MODEL,
        candidate_k: int = 10,
        model_path: Path | None = None,
        threads: int = 8,
    ):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.base = base
        self.candidate_k = candidate_k
        local = model_path or LOCAL_MODEL_DIR
        kwargs = {"specific_model_path": str(local)} if local.is_dir() else {}
        self.model = TextCrossEncoder(model_name=model_name, threads=threads, **kwargs)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        candidates = self.base.search(query, top_k=self.candidate_k)
        if not candidates:
            return []
        scores = list(
            self.model.rerank(query, [c["text"] for c in candidates], batch_size=16)
        )
        for rank, (chunk, score) in enumerate(zip(candidates, scores), 1):
            chunk["first_stage_rank"] = rank
            chunk["first_stage_score"] = chunk.get("score")
            chunk["score"] = float(score)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]

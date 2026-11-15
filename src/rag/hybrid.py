"""BM25 + dense fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = 60,
    weights: list[float] | None = None,
    top_k: int = 20,
) -> list[dict]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, hit in enumerate(ranked, 1):
            cid = hit["chunk_id"]
            chunks.setdefault(cid, hit)
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    fused = []
    for cid in order:
        item = dict(chunks[cid])
        item["score"] = scores[cid]
        fused.append(item)
    return fused


class HybridRetriever:
    def __init__(
        self,
        candidate_k: int = 50,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.weights = [bm25_weight, dense_weight]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        lexical = self.bm25.search(query, top_k=self.candidate_k)
        semantic = self.dense.search(query, top_k=self.candidate_k)
        return rrf_fuse(
            [lexical, semantic],
            k=self.rrf_k,
            weights=self.weights,
            top_k=top_k,
        )

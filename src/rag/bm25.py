"""BM25 lexical retrieval over the guideline chunks."""

from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from rag.corpus import load_chunks


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or load_chunks()
        self.corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:top_k]
        results = []
        for idx in order:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results

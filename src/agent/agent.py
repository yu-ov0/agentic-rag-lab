"""The agent loop: decompose -> retrieve per sub-question -> judge -> retry/refuse.

    question
       │ decompose (rule-based, no LLM)
       ▼
    sub-question 1 ──► retrieve top-k ──► rerank score >= threshold?
       │                                        │ no, and retries left
       │                                        ▼
       │                                     rewrite ──► retrieve again
       ▼
    sub-question 2 ──► ...
       │
       ▼
    merge evidence (dedupe, cap budget) ──► answer with citations
                                         └─► refuse if nothing is sufficient

The baseline in eval/ is the same retriever without the loop: one retrieval on
the whole question, then top-5 evidence. Same models, same budget — the only
difference is decomposition + sufficiency.
"""

from __future__ import annotations

import time
from typing import Protocol

from agent.answer import compose
from agent.decompose import decompose
from agent.models import AgentAnswer, Evidence, RetrievalCall, SubQueryTrace
from agent.sufficiency import DEFAULT_THRESHOLD, is_sufficient, rewrite


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict]: ...


class AgenticRAG:
    def __init__(
        self,
        retriever: Retriever,
        threshold: float = DEFAULT_THRESHOLD,
        per_query_k: int = 5,
        max_evidence: int = 10,
        max_rewrites: int = 1,
        decomposer=decompose,
    ):
        self.retriever = retriever
        self.threshold = threshold
        self.per_query_k = per_query_k
        self.max_evidence = max_evidence
        self.max_rewrites = max_rewrites
        self.decomposer = decomposer

    def ask(self, question: str) -> AgentAnswer:
        started = time.perf_counter()
        sub_questions = self.decomposer(question)

        traces: list[SubQueryTrace] = []
        pool: dict[str, dict] = {}
        calls = 0
        rewrites = 0

        for sub_question in sub_questions:
            trace = SubQueryTrace(sub_question=sub_question)
            hits: list[dict] = []
            for attempt in range(self.max_rewrites + 1):
                query = sub_question if attempt == 0 else rewrite(sub_question, attempt)
                if attempt > 0:
                    rewrites += 1
                t0 = time.perf_counter()
                hits = self.retriever.search(query, top_k=self.per_query_k)
                elapsed = (time.perf_counter() - t0) * 1000
                calls += 1
                ok = is_sufficient(hits, self.threshold)
                trace.calls.append(
                    RetrievalCall(
                        query=query,
                        attempt=attempt,
                        hits=len(hits),
                        hit_ids=[h["chunk_id"] for h in hits],
                        top_score=round(float(hits[0]["score"]), 4) if hits else None,
                        sufficient=ok,
                        latency_ms=round(elapsed, 2),
                    )
                )
                if ok:
                    break

            trace.sufficient = is_sufficient(hits, self.threshold)
            if trace.sufficient:
                hits = sorted(hits, key=lambda h: float(h["score"]), reverse=True)
                for hit in hits:
                    prev = pool.get(hit["chunk_id"])
                    if prev is None or float(hit["score"]) > float(prev["score"]):
                        pool[hit["chunk_id"]] = hit
                trace.evidence_ids = [h["chunk_id"] for h in hits]
            traces.append(trace)

        ranked_pool = sorted(
            pool.values(), key=lambda h: float(h["score"]), reverse=True
        )
        evidence = ranked_pool[: self.max_evidence]
        kept = {e["chunk_id"] for e in evidence}
        for trace in traces:
            trace.evidence_ids = [cid for cid in trace.evidence_ids if cid in kept]

        answer_text, citations = compose(question, traces, evidence)
        answered = sum(1 for t in traces if t.sufficient)
        refused = answered == 0
        reason = None
        if refused:
            reason = (
                "insufficient_evidence"
                if calls and any(t.calls[-1].hits for t in traces)
                else "no_candidates"
            )

        return AgentAnswer(
            question=question,
            answer=answer_text,
            refused=refused,
            partial=0 < answered < len(traces),
            reason=reason,
            citations=citations,
            sub_questions=traces,
            evidence=[
                Evidence(
                    chunk_id=e["chunk_id"],
                    doc_id=e["doc_id"],
                    page=e["page"],
                    score=round(float(e["score"]), 4),
                    text=e["text"],
                )
                for e in evidence
            ],
            pool=[
                Evidence(
                    chunk_id=e["chunk_id"],
                    doc_id=e["doc_id"],
                    page=e["page"],
                    score=round(float(e["score"]), 4),
                    text=e["text"],
                )
                for e in ranked_pool
            ],
            retrieval_calls=calls,
            rewrites=rewrites,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

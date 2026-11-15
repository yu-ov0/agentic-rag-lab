"""Typed trace objects. Every decision the agent makes is recorded here.

The trace is the product: when an answer is wrong you need to see which
sub-question failed, whether it was rewritten, and what score the reranker
gave. Without that, "Agentic RAG" is just a longer prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalCall(BaseModel):
    query: str
    attempt: int  # 0 = original sub-question, 1..n = rewrite
    hits: int
    hit_ids: list[str] = Field(default_factory=list)
    top_score: float | None = None
    sufficient: bool = False
    latency_ms: float = 0.0


class SubQueryTrace(BaseModel):
    sub_question: str
    sufficient: bool = False
    calls: list[RetrievalCall] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    score: float
    text: str


class AgentAnswer(BaseModel):
    question: str
    answer: str
    refused: bool = False
    partial: bool = False
    reason: str | None = None
    citations: list[dict] = Field(default_factory=list)
    sub_questions: list[SubQueryTrace] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    # Every deduped candidate before the evidence budget is applied. The eval
    # uses it to compare merge policies without paying for another retrieval.
    pool: list[Evidence] = Field(default_factory=list)
    retrieval_calls: int = 0
    rewrites: int = 0
    latency_ms: float = 0.0

"""The agent loop is tested with a stub retriever, so CI needs no models.

The stub lets each test control the rerank score, which is the only thing the
agent uses to decide "sufficient / rewrite / refuse".
"""

from __future__ import annotations

from agent.agent import AgenticRAG
from agent.decompose import decompose
from agent.sufficiency import is_sufficient, rewrite


class StubRetriever:
    """Returns chunks with a per-query score; chunk ids can be forced."""

    def __init__(
        self,
        scores: dict[str, float],
        default_score: float = 0.0,
        chunk_prefix: str | None = None,
        n_hits: int = 1,
    ):
        self.scores = scores
        self.default_score = default_score
        self.chunk_prefix = chunk_prefix
        self.n_hits = n_hits
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        self.queries.append(query)
        score = self.scores.get(query, self.default_score)
        if score == 0.0:
            return []
        prefix = self.chunk_prefix or f"doc{abs(hash(query)) % 1000}"
        return [
            {
                "chunk_id": f"{prefix}:c{i}",
                "doc_id": "hypertension_2024",
                "page": 1,
                "text": f"关于{query}的指南原文。",
                "score": score - i * 0.1,
            }
            for i in range(min(self.n_hits, top_k))
        ]


def test_decompose_splits_two_questions():
    parts = decompose("诊室血压诊断界值是多少？糖尿病的诊断标准是什么？")
    assert parts == ["诊室血压诊断界值是多少？", "糖尿病的诊断标准是什么？"]


def test_decompose_keeps_single_question():
    assert decompose("高血压患者每日食盐摄入量应控制在多少？") == [
        "高血压患者每日食盐摄入量应控制在多少？"
    ]


def test_decompose_dedupes_identical_sub_questions():
    parts = decompose("低血糖如何分级？低血糖如何分级？")
    assert parts == ["低血糖如何分级？"]


def test_sufficiency_uses_top_score():
    assert is_sufficient([{"score": 2.36}], 2.0)
    assert not is_sufficient([{"score": 1.53}], 2.0)
    assert not is_sufficient([], 2.0)


def test_rewrite_strips_fillers_then_adds_context():
    first = rewrite("诊室血压诊断高血压的界值是多少？", 1)
    assert "多少" not in first and "界值" in first
    second = rewrite("诊室血压诊断高血压的界值是多少？", 2)
    assert "指南" in second


def test_agent_answers_both_hops_with_two_calls():
    retriever = StubRetriever(
        {"诊室血压诊断界值是多少？": 3.0, "糖尿病的诊断标准是什么？": 2.5}
    )
    result = AgenticRAG(retriever, threshold=2.0).ask(
        "诊室血压诊断界值是多少？糖尿病的诊断标准是什么？"
    )
    assert not result.refused
    assert len(result.sub_questions) == 2
    assert result.retrieval_calls == 2
    assert result.rewrites == 0
    assert len(result.evidence) == 2
    assert "[1]" in result.answer and "[2]" in result.answer


def test_agent_refuses_when_nothing_is_sufficient():
    retriever = StubRetriever({}, default_score=0.5)
    result = AgenticRAG(retriever, threshold=2.0, max_rewrites=1).ask(
        "肺癌靶向药的推荐剂量是多少？"
    )
    assert result.refused
    assert result.reason == "insufficient_evidence"
    # original + one rewrite
    assert result.retrieval_calls == 2
    assert result.rewrites == 1


def test_agent_retries_with_rewrite_and_recovers():
    question = "诊室血压诊断高血压的界值是多少？"
    stripped = rewrite(question, 1)
    retriever = StubRetriever({stripped: 2.6})
    result = AgenticRAG(retriever, threshold=2.0, max_rewrites=1).ask(question)
    assert not result.refused
    assert result.retrieval_calls == 2
    assert result.rewrites == 1
    assert result.sub_questions[0].calls[0].sufficient is False
    assert result.sub_questions[0].calls[1].sufficient is True


def test_agent_marks_partial_when_one_hop_fails():
    retriever = StubRetriever({"诊室血压诊断界值是多少？": 3.0}, default_score=0.4)
    result = AgenticRAG(retriever, threshold=2.0, max_rewrites=0).ask(
        "诊室血压诊断界值是多少？糖尿病的诊断标准是什么？"
    )
    assert not result.refused
    assert result.partial
    assert "未找到足够依据" in result.answer


def test_agent_dedupes_shared_chunks():
    # Both hops hit the same chunk: evidence must not repeat it.
    retriever = StubRetriever(
        {"A 问题是什么？": 3.0, "B 问题是什么？": 3.0}, chunk_prefix="docX"
    )
    result = AgenticRAG(retriever, threshold=2.0).ask("A 问题是什么？B 问题是什么？")
    ids = [e.chunk_id for e in result.evidence]
    assert ids == ["docX:c0"]


def test_agent_caps_evidence_budget():
    retriever = StubRetriever({"A 问题是什么？": 3.0, "B 问题是什么？": 3.0}, n_hits=3)
    result = AgenticRAG(retriever, threshold=2.0, max_evidence=4).ask(
        "A 问题是什么？B 问题是什么？"
    )
    assert len(result.evidence) == 4


def test_citation_indices_are_in_range():
    from agent.answer import validate_citations

    retriever = StubRetriever(
        {"诊室血压诊断界值是多少？": 3.0, "糖尿病的诊断标准是什么？": 2.5}
    )
    result = AgenticRAG(retriever, threshold=2.0).ask(
        "诊室血压诊断界值是多少？糖尿病的诊断标准是什么？"
    )
    assert validate_citations(result.answer, len(result.evidence)) == []

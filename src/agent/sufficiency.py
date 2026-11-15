"""Sufficiency check and query rewriting.

Sufficiency uses the cross-encoder score of the best chunk. It is a direct
(query, chunk) relevance judgement, so a low score means "retrieval did not
find the passage that answers this sub-question" — which is exactly the signal
the agent needs to either rewrite or refuse.
"""

from __future__ import annotations

import re

# Calibrated on med-rag-lab's 26 single-hop questions: the lowest answerable
# top-1 rerank score was 2.361, the highest out-of-corpus score was 1.532.
DEFAULT_THRESHOLD = 2.0

_FILLERS = (
    "是多少",
    "是什么",
    "有哪些",
    "怎么样",
    "为什么",
    "什么",
    "多少",
    "哪些",
    "如何",
    "请问",
    "吗",
    "呢",
    "？",
    "?",
)
_DOMAIN_HINT = "临床指南 推荐"


def is_sufficient(hits: list[dict], threshold: float = DEFAULT_THRESHOLD) -> bool:
    return bool(hits) and float(hits[0].get("score", 0.0)) >= threshold


def rewrite(query: str, attempt: int = 1) -> str:
    """Attempt 1 strips interrogative fillers; attempt 2 adds domain context."""
    text = query
    for filler in _FILLERS:
        text = text.replace(filler, "")
    text = re.sub(r"\s+", " ", text).strip(" ，,。")
    if attempt >= 2:
        text = f"{text} {_DOMAIN_HINT}".strip()
    return text or query

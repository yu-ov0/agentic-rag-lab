"""Rule-based question decomposition.

A multi-hop question is written as several complete questions ("A？B？"). The
decomposer splits on sentence-final question marks, which is deterministic and
auditable — no LLM call, no extra latency, and the split can be asserted in a
test.

Known limits (stated in the README, not hidden):
- a single sentence that hides two hops ("A 和 B 分别是什么？") is not split;
- an anaphoric follow-up ("那它的剂量呢？") cannot be resolved without history.
"""

from __future__ import annotations

import re

# Split after a question mark and keep the mark: a sub-question is then
# byte-identical to the single-hop question it came from, so the multi-hop
# numbers stay comparable with the single-hop baseline.
_SPLIT = re.compile(r"[^？?]+[？?]?")
# "；" is a real clause boundary in Chinese guidelines, unlike "，" or "、".
_CLAUSE = re.compile(r"[；;]")
MIN_LEN = 6


def decompose(question: str) -> list[str]:
    question = question.strip()
    parts = [
        m.group(0).strip() for m in _SPLIT.finditer(question) if m.group(0).strip()
    ]

    expanded: list[str] = []
    for part in parts:
        if len(part) > 40 and _CLAUSE.search(part):
            expanded.extend(p.strip() for p in _CLAUSE.split(part) if p.strip())
        else:
            expanded.append(part)

    kept = [p for p in expanded if len(p) >= MIN_LEN]
    if not kept:
        return [question]
    # De-duplicate while preserving order: the same question twice adds a
    # retrieval call but no evidence.
    seen: set[str] = set()
    unique = []
    for part in kept:
        if part not in seen:
            seen.add(part)
            unique.append(part)
    return unique

"""Extractive answer composition with citations.

No LLM in the default path: every sentence in the answer is copied from a
retrieved chunk and carries an index. That makes the evaluation about
retrieval quality instead of generation fluency, and it means the project can
be run and tested offline.
"""

from __future__ import annotations

import re

DOC_TITLES = {
    "hypertension_2024": "《中国高血压防治指南（2024年修订版）》",
    "t2dm_2020": "《中国2型糖尿病防治指南（2020年版）》",
}


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


# Extractive answers are mostly "the sentence with the numbers". Without this
# bonus the picker prefers a sentence that repeats the question's words
# ("...诊断标准亦可参见表7") over the sentence that actually states the value
# ("非同日3次测量诊室血压≥140/90 mmHg").
_UNIT = re.compile(r"(mmHg|mmol|mg|ml|g/d|%|℃|岁|次/|h\b|d\b)")
_DIGIT = re.compile(r"\d")


def _sentence_score(sentence: str, target: set[str]) -> float:
    score = len(_bigrams(sentence) & target)
    has_digit = bool(_DIGIT.search(sentence))
    has_unit = bool(_UNIT.search(sentence))
    if has_digit and has_unit:
        score += 4
    elif has_digit:
        score += 1
    return score


def best_sentence(text: str, query: str, limit: int = 180) -> str:
    """Pick the sentence with the best keyword overlap plus a numeric bonus."""
    sentences = [s.strip() for s in re.split(r"[。；\n]", text) if len(s.strip()) >= 8]
    if not sentences:
        return text.strip()[:limit]
    target = _bigrams(query)
    best = max(sentences, key=lambda s: _sentence_score(s, target))
    return best[:limit]


def citation(index: int, chunk: dict) -> dict:
    return {
        "index": index,
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "doc_title": DOC_TITLES.get(chunk["doc_id"], chunk["doc_id"]),
        "page": chunk["page"],
        "text": chunk["text"],
    }


def compose(question: str, traces, evidence: list[dict]) -> tuple[str, list[dict]]:
    """One bullet per answered sub-question, each with a source index."""
    index_of = {e["chunk_id"]: i for i, e in enumerate(evidence, 1)}
    citations = [citation(i, e) for i, e in enumerate(evidence, 1)]

    lines = []
    for trace in traces:
        if not trace.sufficient or not trace.evidence_ids:
            continue
        best_id = trace.evidence_ids[0]
        if best_id not in index_of:
            continue
        chunk = next(e for e in evidence if e["chunk_id"] == best_id)
        excerpt = best_sentence(chunk["text"], trace.sub_question)
        lines.append(f"- {trace.sub_question}：{excerpt} [{index_of[best_id]}]")

    missing = [t.sub_question for t in traces if not t.sufficient]
    if not lines:
        return "", citations

    body = "根据指南资料：\n" + "\n".join(lines)
    if missing:
        body += "\n未找到足够依据的子问题：" + "；".join(missing)
    return body, citations


def validate_citations(text: str, n_sources: int) -> list[int]:
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in cited if i < 1 or i > n_sources)

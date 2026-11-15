"""Baseline vs agentic on the 12 multi-hop questions.

baseline : one retrieval on the whole question -> rerank top-5 evidence
agentic  : decompose -> top-5 per sub-question -> merge -> answer or refuse

Three evidence sets, all from the same retrieval calls:
- baseline top-5
- agentic, budget-matched (first 5 of the merged evidence)
- agentic, full budget (up to 10, i.e. 5 per hop)

Also reported:
- per-hop ceiling: could the gold chunk be found at all when the hop is
  retrieved on its own? If agentic == ceiling, the loop itself loses nothing
  and the remaining misses are single-hop retrieval quality, not decomposition.
- phrase coverage: same idea, but the answer phrase may live in any chunk.
- refusal rate on 10 out-of-corpus questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.agent import AgenticRAG
from rag.hybrid import HybridRetriever
from rag.rerank import Reranker

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
UNANSWERABLE = ROOT / "eval" / "unanswerable.jsonl"
RESULTS = ROOT / "eval" / "results"
CANDIDATE_K = 10
BASELINE_K = 5
PER_QUERY_K = 5
MAX_EVIDENCE = 10
BUDGET = 5


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_retriever(candidate_k: int = CANDIDATE_K) -> Reranker:
    return Reranker(HybridRetriever(), candidate_k=candidate_k)


def recall(evidence_ids: list[str], gold: list[str]) -> tuple[float, float]:
    hits = sum(1 for g in gold if g in evidence_ids)
    return (1.0 if hits == len(gold) else 0.0), hits / len(gold)


def phrase_recall(texts: list[str], phrases: list[str]) -> tuple[float, float]:
    """Coverage metric: is the answer phrase present, whichever chunk carries it?

    A single-chunk gold label is brittle for guidelines, where the same
    recommendation is repeated in the key-points box and in the chapter body.
    """
    hits = sum(1 for p in phrases if any(p in t for t in texts))
    return (1.0 if hits == len(phrases) else 0.0), hits / len(phrases)


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def mean(rows: list[dict], *path: str) -> float:
    total = 0.0
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        total += node
    return total / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="debug: first N rows")
    args = parser.parse_args()

    rows = load(EVAL_SET)
    if args.limit:
        rows = rows[: args.limit]

    retriever = build_retriever()
    agent = AgenticRAG(
        retriever, per_query_k=PER_QUERY_K, max_evidence=MAX_EVIDENCE, max_rewrites=1
    )

    out_rows = []
    for row in rows:
        gold = row["gold_chunk_ids"]
        phrases = row["gold_phrases"]

        t0 = time.perf_counter()
        baseline_hits = retriever.search(row["question"], top_k=BASELINE_K)
        baseline_ms = (time.perf_counter() - t0) * 1000
        baseline_ids = [h["chunk_id"] for h in baseline_hits]

        result = agent.ask(row["question"])
        agent_ids = [e.chunk_id for e in result.evidence]
        agent_texts = [e.text for e in result.evidence]

        # Could each hop's gold be retrieved on its own? calls[0] is the
        # unmodified sub-question, so no extra retrieval is needed.
        ceiling = all(
            g in trace.calls[0].hit_ids for trace, g in zip(result.sub_questions, gold)
        )

        b_all, b_part = recall(baseline_ids, gold)
        a_all, a_part = recall(agent_ids, gold)
        a5_all, a5_part = recall(agent_ids[:BUDGET], gold)
        bp_all, _ = phrase_recall([h["text"] for h in baseline_hits], phrases)
        ap_all, _ = phrase_recall(agent_texts, phrases)

        out_rows.append(
            {
                "qid": row["qid"],
                "type": row["type"],
                "question": row["question"],
                "gold_chunk_ids": gold,
                "hop_ceiling": 1.0 if ceiling else 0.0,
                "baseline": {
                    "evidence_recall": b_all,
                    "gold_hit_rate": b_part,
                    "phrase_recall": bp_all,
                    "evidence_ids": baseline_ids,
                    "calls": 1,
                    "latency_ms": round(baseline_ms, 2),
                },
                "agentic": {
                    "evidence_recall": a_all,
                    "gold_hit_rate": a_part,
                    "evidence_recall_budget5": a5_all,
                    "gold_hit_rate_budget5": a5_part,
                    "phrase_recall": ap_all,
                    "evidence_ids": agent_ids,
                    "n_evidence": len(agent_ids),
                },
                "trace": {
                    "sub_questions": [t.sub_question for t in result.sub_questions],
                    "sufficient": [t.sufficient for t in result.sub_questions],
                    "rewrites": result.rewrites,
                    "calls": result.retrieval_calls,
                    "refused": result.refused,
                    "partial": result.partial,
                    "latency_ms": result.latency_ms,
                },
            }
        )
        print(
            f"{row['qid']} {row['type']:<9} "
            f"base={b_all:.0f} agent={a_all:.0f}(@5 {a5_all:.0f}) "
            f"ceiling={int(ceiling)} ev={len(agent_ids)} "
            f"calls={result.retrieval_calls} rw={result.rewrites}"
        )

    for kind in ("cross_doc", "same_doc"):
        subset = [r for r in out_rows if r["type"] == kind]
        if not subset:
            continue
        print(
            f"{kind:<9} n={len(subset)} "
            f"base={mean(subset, 'baseline', 'evidence_recall'):.3f} "
            f"agent={mean(subset, 'agentic', 'evidence_recall'):.3f} "
            f"agent@5={mean(subset, 'agentic', 'evidence_recall_budget5'):.3f} "
            f"ceiling={mean(subset, 'hop_ceiling'):.3f}"
        )

    summary = {
        "n": len(out_rows),
        "candidate_k": CANDIDATE_K,
        "per_query_k": PER_QUERY_K,
        "max_evidence": MAX_EVIDENCE,
        "budget": BUDGET,
        "baseline": {
            "evidence_recall": mean(out_rows, "baseline", "evidence_recall"),
            "gold_hit_rate": mean(out_rows, "baseline", "gold_hit_rate"),
            "phrase_recall": mean(out_rows, "baseline", "phrase_recall"),
            "calls": 1,
            "latency_p50_ms": pct([r["baseline"]["latency_ms"] for r in out_rows], 50),
            "latency_p95_ms": pct([r["baseline"]["latency_ms"] for r in out_rows], 95),
        },
        "agentic": {
            "evidence_recall": mean(out_rows, "agentic", "evidence_recall"),
            "evidence_recall_budget5": mean(
                out_rows, "agentic", "evidence_recall_budget5"
            ),
            "gold_hit_rate": mean(out_rows, "agentic", "gold_hit_rate"),
            "gold_hit_rate_budget5": mean(out_rows, "agentic", "gold_hit_rate_budget5"),
            "phrase_recall": mean(out_rows, "agentic", "phrase_recall"),
            "avg_evidence": mean(out_rows, "agentic", "n_evidence"),
            "avg_calls": mean(out_rows, "trace", "calls"),
            "rewrites": sum(r["trace"]["rewrites"] for r in out_rows),
            "refused": sum(1 for r in out_rows if r["trace"]["refused"]),
            "partial": sum(1 for r in out_rows if r["trace"]["partial"]),
            "latency_p50_ms": pct([r["trace"]["latency_ms"] for r in out_rows], 50),
            "latency_p95_ms": pct([r["trace"]["latency_ms"] for r in out_rows], 95),
        },
        "hop_ceiling": mean(out_rows, "hop_ceiling"),
    }

    negatives = load(UNANSWERABLE)
    refusals = []
    for row in negatives:
        result = agent.ask(row["question"])
        refusals.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "refused": result.refused,
                "top_scores": [
                    t.calls[0].top_score for t in result.sub_questions if t.calls
                ],
                "calls": result.retrieval_calls,
            }
        )
    summary["refusal"] = {
        "n": len(refusals),
        "refusal_rate": sum(r["refused"] for r in refusals) / len(refusals),
        "rows": refusals,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "agentic_vs_baseline.json").write_text(
        json.dumps(
            {"summary": summary, "rows": out_rows}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

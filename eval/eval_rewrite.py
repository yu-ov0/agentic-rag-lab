"""Does the rewrite step actually recover anything?

The multi-hop set never triggers a rewrite: every sub-question is a verbatim
guideline question, so the reranker is already confident. This set is the
opposite — colloquial, patient-style wording. We measure how many questions
pass the sufficiency threshold before and after each rewrite attempt.

This is the honest way to report a retry mechanism: if it recovers nothing,
the number says so and the feature should be cut, not described.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import build_retriever

from agent.sufficiency import DEFAULT_THRESHOLD, is_sufficient, rewrite

QUERIES = ROOT / "eval" / "colloquial.jsonl"
RESULTS = ROOT / "eval" / "results" / "rewrite_recovery.json"
TOP_K = 5


def main() -> None:
    retriever = build_retriever()
    rows = [json.loads(line) for line in QUERIES.open(encoding="utf-8") if line.strip()]

    for row in rows:
        row["attempts"] = []
        for attempt in range(3):
            query = (
                row["question"] if attempt == 0 else rewrite(row["question"], attempt)
            )
            hits = retriever.search(query, top_k=TOP_K)
            ids = [h["chunk_id"] for h in hits]
            row["attempts"].append(
                {
                    "attempt": attempt,
                    "query": query,
                    "top_score": round(float(hits[0]["score"]), 4),
                    "sufficient": is_sufficient(hits, DEFAULT_THRESHOLD),
                    "gold_in_topk": row["gold_chunk_id"] in ids,
                }
            )
        marks = " ".join(
            f"{a['top_score']:5.2f}{'*' if a['sufficient'] else ' '}"
            for a in row["attempts"]
        )
        print(f"{row['qid']} {row['question'][:24]:<26} {marks}")

    n = len(rows)
    summary = {
        "n": n,
        "threshold": DEFAULT_THRESHOLD,
        "top_k": TOP_K,
        "answerable_at": {
            f"attempt_{a}": sum(r["attempts"][a]["sufficient"] for r in rows) / n
            for a in range(3)
        },
        "gold_in_topk_at": {
            f"attempt_{a}": sum(r["attempts"][a]["gold_in_topk"] for r in rows) / n
            for a in range(3)
        },
        "recovered_by_rewrite": [
            r["qid"]
            for r in rows
            if not r["attempts"][0]["sufficient"]
            and any(a["sufficient"] for a in r["attempts"][1:])
        ],
    }
    RESULTS.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

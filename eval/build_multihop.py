"""Compose the multi-hop evaluation set from labelled single-hop questions.

Each row is two complete questions joined into one string, so a rule-based
decomposer can split it deterministically. Both gold chunks are kept, and the
pairs are chosen so the two gold chunks differ — otherwise "two hops" would
mean "one passage answers both".

6 pairs cross the two guidelines, 6 stay inside one guideline (the hard case:
same document, same vocabulary, different passages).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "eval" / "single_hop.jsonl"
OUT = ROOT / "eval" / "eval_set.jsonl"

# (left qid, right qid). Verified: every pair has two distinct gold chunks.
PAIRS = [
    ("h01", "d01"),  # cross-doc: 诊室血压界值 + 糖尿病诊断标准
    ("h08", "d08"),  # cross-doc: 食盐摄入 + 营养治疗原则
    ("h07", "d04"),  # cross-doc: 高龄降压目标 + 综合控制目标
    ("h12", "d10"),  # cross-doc: 危险分层 + 糖尿病肾病筛查
    ("h09", "d12"),  # cross-doc: 联合治疗时机 + SGLT2i/GLP-1RA 时机
    ("h13", "d06"),  # cross-doc: 合并糖尿病降压目标 + SGLT2i 机制
    ("h04", "h12"),  # same-doc: 血压分级 + 危险分层
    ("h05", "h14"),  # same-doc: 一般降压目标 + 启动药物时机
    ("h10", "h08"),  # same-doc: 白大衣性高血压 + 食盐摄入
    ("d06", "d07"),  # same-doc: SGLT2i 机制 + GLP-1RA 作用
    ("d09", "d11"),  # same-doc: 低血糖分级 + 视网膜病变筛查
    ("d05", "d08"),  # same-doc: 二甲双胍地位 + 医学营养治疗
]


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["qid"]: row for row in map(json.loads, f)}


def main() -> None:
    singles = load(SINGLE)
    rows = []
    for i, (left_id, right_id) in enumerate(PAIRS, 1):
        left, right = singles[left_id], singles[right_id]
        gold = [left["gold_chunk_id"], right["gold_chunk_id"]]
        assert len(set(gold)) == 2, f"{left_id}+{right_id} share a gold chunk"
        cross_doc = left["doc_id"] != right["doc_id"]
        rows.append(
            {
                "qid": f"mh{i:02d}",
                "question": left["question"] + right["question"],
                "sub_questions": [left["question"], right["question"]],
                "gold_chunk_ids": gold,
                "gold_phrases": [left["gold_phrase"], right["gold_phrase"]],
                "gold_doc_ids": [left["doc_id"], right["doc_id"]],
                "type": "cross_doc" if cross_doc else "same_doc",
            }
        )

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUT} ({len(rows)} rows, "
        f"{sum(r['type'] == 'cross_doc' for r in rows)} cross-doc)"
    )


if __name__ == "__main__":
    main()

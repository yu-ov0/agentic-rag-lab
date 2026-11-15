"""Corpus loading shared by every retriever.

The chunks come from med-rag-lab's ingestion (PyMuPDF -> 450-char windows with
80-char overlap, page numbers preserved). This project reuses that corpus on
purpose: the question here is not "can I parse PDFs" but "does decomposing a
multi-hop question actually retrieve better evidence".
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_index(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}

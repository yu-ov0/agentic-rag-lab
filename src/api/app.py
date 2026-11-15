"""Minimal HTTP surface: one question in, answer + full trace out."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.agent import AgenticRAG
from rag.hybrid import HybridRetriever
from rag.rerank import Reranker

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Building the retriever loads the ONNX models and the cached corpus
    # matrix; doing it once at startup keeps /ask fast.
    STATE["agent"] = AgenticRAG(Reranker(HybridRetriever(), candidate_k=10))
    yield
    STATE.clear()


app = FastAPI(title="agentic-rag-lab", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "agent_ready": "agent" in STATE}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return STATE["agent"].ask(req.question).model_dump()

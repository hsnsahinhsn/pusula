"""Pusula HTTP API (FastAPI). Request/answer shapes follow vllm-sr/d3's /v1/systemone.

    # Pusula (Turkish Decision 3.0): base d3 snapshot + Pusula adapter
    PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula-d3 pusula-serve
    # Pusula Nano (TabiBERT)
    PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve

Endpoints:
    POST /v1/systemone        {"state": str, "questions": {name: {type, instructions, criteria}}}
    POST /v1/systemone/batch  {"requests": [ {state, questions}, ... ]}
    POST /v1/route            {"state": str, "universal": true}  -> taxonomy routing
    GET  /v1/taxonomy         built-in taxonomy
    GET  /health              liveness (always 200)
    GET  /ready               503 until the model is loaded
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .schema import Question
from .taxonomy import load_taxonomy

MAX_STATE_CHARS = int(os.getenv("PUSULA_MAX_STATE_CHARS", "60000"))
MAX_QUESTIONS = int(os.getenv("PUSULA_MAX_QUESTIONS", "32"))
MAX_BATCH = int(os.getenv("PUSULA_MAX_BATCH", "64"))


class DecideRequest(BaseModel):
    state: str = Field(..., min_length=1, max_length=MAX_STATE_CHARS)
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _limit(cls, v):
        if not 1 <= len(v) <= MAX_QUESTIONS:
            raise ValueError(f"1-{MAX_QUESTIONS} questions per request")
        return v


class BatchRequest(BaseModel):
    requests: list[DecideRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class RouteRequest(BaseModel):
    state: str = Field(..., min_length=1, max_length=MAX_STATE_CHARS)
    universal: bool = True


def create_app(engine: Any = None, model_dir: Optional[str] = None) -> FastAPI:
    """`engine` may be injected (tests); otherwise it is loaded from `model_dir` / PUSULA_MODEL_DIR."""
    state: dict[str, Any] = {"engine": engine}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_dir or os.getenv("PUSULA_MODEL_DIR")
        d3_base = os.getenv("PUSULA_D3_BASE")
        if state["engine"] is None and d3_base:
            from .inference import D3Engine

            state["engine"] = D3Engine.load(d3_base, adapter=path, device=os.getenv("PUSULA_DEVICE") or "cuda:0")
        elif state["engine"] is None and path:
            from .inference import DecisionEngine

            state["engine"] = DecisionEngine.load(path, device=os.getenv("PUSULA_DEVICE"))
        yield

    app = FastAPI(title="Pusula Decision API", version=__version__, lifespan=lifespan)
    origins = [o.strip() for o in os.getenv("PUSULA_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    def get_engine():
        if state["engine"] is None:
            raise HTTPException(503, "Model yüklü değil (PUSULA_D3_BASE / PUSULA_MODEL_DIR ayarlayın).")
        return state["engine"]

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "model_loaded": state["engine"] is not None}

    @app.get("/ready")
    def ready():
        get_engine()
        return {"status": "ready"}

    @app.get("/v1/taxonomy")
    def taxonomy():
        return load_taxonomy().to_dict()

    # Plain `def` handlers run in FastAPI's threadpool, so inference never blocks the event loop.
    @app.post("/v1/systemone")
    def systemone(req: DecideRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        answers = engine.decide(req.state, req.questions)
        return {"answers": answers, "model": engine.model_name,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/v1/systemone/batch")
    def systemone_batch(req: BatchRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        results = engine.decide_many([(r.state, r.questions) for r in req.requests])
        return {"results": [{"answers": a} for a in results], "model": engine.model_name,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/v1/route")
    def route(req: RouteRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        out = engine.route(req.state, universal=req.universal)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    return app


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Pusula API sunucusu")
    p.add_argument("--model", default=os.getenv("PUSULA_MODEL_DIR"), help="checkpoint dizini")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    uvicorn.run(create_app(model_dir=a.model), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

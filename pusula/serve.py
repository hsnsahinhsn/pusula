"""Pusula HTTP API (FastAPI). Request/answer shapes follow vllm-sr/d3's /v1/systemone.

    # Pusula (Turkish Decision 3.0): base d3 snapshot + Pusula adapter
    PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula-d3 pusula-serve
    # Pusula Nano (TabiBERT)
    PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve

Endpoints:
    POST /v1/systemone        {"state": str, "questions": {name: {type, instructions, criteria}}}
    POST /v1/systemone/batch  {"requests": [ {state, questions}, ... ]}
    POST /v1/route            {"state": str, "universal": true}  -> taxonomy routing
                              {"state": str, "topics": {key: description | {"description", "intents"}}}
                              -> routing through the institution's own list (konu -> niyet)
    GET  /v1/taxonomy         built-in taxonomy
    GET  /health              liveness (always 200)
    GET  /ready               503 until the model is loaded

Listens on 127.0.0.1 unless --host / PUSULA_HOST says otherwise. With PUSULA_API_KEY set, every /v1/* request needs
`Authorization: Bearer <key>` (or `X-API-Key: <key>`); set it before exposing the server to a network.
"""
from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .schema import Question
from .taxonomy import load_taxonomy, topic_questions

MAX_STATE_CHARS = int(os.getenv("PUSULA_MAX_STATE_CHARS", "60000"))
MAX_QUESTIONS = int(os.getenv("PUSULA_MAX_QUESTIONS", "32"))
MAX_BATCH = int(os.getenv("PUSULA_MAX_BATCH", "64"))
MAX_SCAN_CHARS = int(os.getenv("PUSULA_MAX_SCAN_CHARS", "200000"))


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
    # An institution's own routing list (konu -> niyet); without it the built-in taxonomy is used.
    topics: Optional[dict[str, Any]] = None

    @field_validator("topics")
    @classmethod
    def _valid_topics(cls, v):
        if v is not None:
            topic_questions(v)  # raises ValueError (-> 422) for an invalid list
        return v


class ScanRequest(DecideRequest):
    """A long text (web page, document, e-mail thread) checked part by part; see BaseEngine.scan."""
    state: str = Field(..., min_length=1, max_length=MAX_SCAN_CHARS)
    window_chars: Optional[int] = Field(None, ge=256, le=60000)


def create_app(engine: Any = None, model_dir: Optional[str] = None, api_key: Optional[str] = None) -> FastAPI:
    """`engine` may be injected (tests); otherwise it is loaded from `model_dir` / PUSULA_MODEL_DIR. `api_key`
    (default: PUSULA_API_KEY) protects the /v1/* endpoints."""
    state: dict[str, Any] = {"engine": engine}
    key = api_key if api_key is not None else os.getenv("PUSULA_API_KEY", "")

    def check_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
        if not key:
            return
        given = x_api_key or (authorization[7:] if authorization and authorization.startswith("Bearer ") else "")
        if not hmac.compare_digest(given.encode(), key.encode()):
            raise HTTPException(401, "Geçersiz ya da eksik API anahtarı.")

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
    guarded = [Depends(check_key)]
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

    @app.get("/v1/taxonomy", dependencies=guarded)
    def taxonomy():
        return load_taxonomy().to_dict()

    # Plain `def` handlers run in FastAPI's threadpool, so inference never blocks the event loop.
    @app.post("/v1/systemone", dependencies=guarded)
    def systemone(req: DecideRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        answers = engine.decide(req.state, req.questions)
        return {"answers": answers, "model": engine.model_name,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/v1/systemone/batch", dependencies=guarded)
    def systemone_batch(req: BatchRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        results = engine.decide_many([(r.state, r.questions) for r in req.requests])
        return {"results": [{"answers": a} for a in results], "model": engine.model_name,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/v1/scan", dependencies=guarded)
    def scan(req: ScanRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        out = engine.scan(req.state, req.questions, window_chars=req.window_chars)
        out.update(model=engine.model_name, latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        return out

    @app.post("/v1/route", dependencies=guarded)
    def route(req: RouteRequest):
        engine = get_engine()
        t0 = time.perf_counter()
        out = (engine.route(req.state, universal=req.universal, topics=req.topics) if req.topics is not None
               else engine.route(req.state, universal=req.universal))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    return app


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Pusula API sunucusu")
    p.add_argument("--model", default=os.getenv("PUSULA_MODEL_DIR"), help="checkpoint dizini")
    p.add_argument("--host", default=os.getenv("PUSULA_HOST", "127.0.0.1"),
                   help="0.0.0.0 makes it reachable from the network: set PUSULA_API_KEY first")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    uvicorn.run(create_app(model_dir=a.model), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

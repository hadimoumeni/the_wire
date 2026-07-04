"""FastAPI service for The Wire.

Endpoints:
    GET  /api/health          liveness + whether a local model is loaded
    GET  /api/sample          the bundled sample transcript
    POST /api/analyze         {transcript, ticker, quarter} -> Thesis
    GET  /api/theses          list stored theses (summaries)
    GET  /api/theses/{id}     one full Thesis

Also serves the built Astro frontend (frontend/dist) at / when present, so
`python api.py` gives you the whole demo at http://localhost:2424.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from wire import db
from wire.llm import get_client
from wire.pipeline import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "data", "sample_transcript.txt")
DIST = os.path.join(HERE, "frontend", "dist")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="The Wire", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,
                          lambda r, e: HTTPException(429, "rate limit exceeded"))
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AnalyzeIn(BaseModel):
    transcript: str = Field(min_length=50)
    ticker: str = "N/A"
    quarter: str = "N/A"


@app.get("/api/health")
def health():
    client = get_client()
    return {
        "ok": True,
        "model_available": client is not None,
        "model": client.model if client else "heuristic-baseline",
    }


@app.get("/api/sample")
def sample():
    try:
        return {"transcript": open(SAMPLE, encoding="utf-8").read()}
    except OSError:
        raise HTTPException(404, "sample not found")


@app.post("/api/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request, body: AnalyzeIn):
    thesis = await run_pipeline(
        body.transcript, ticker=body.ticker, quarter=body.quarter, persist=True)
    return thesis.model_dump()


@app.get("/api/theses")
def theses():
    return db.list_theses()


@app.get("/api/theses/{thesis_id}")
def thesis(thesis_id: int):
    t = db.get_thesis(thesis_id)
    if t is None:
        raise HTTPException(404, "not found")
    return t.model_dump()


# Serve the built frontend last, so /api/* routes win.
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "2424")))

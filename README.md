# The Wire

An agentic system that ingests earnings-call transcripts and produces a
**verified, structured investment thesis** — management sentiment, key risk
language, forward guidance, and an overall stance — where **every claim is
grounded in an exact quote from the transcript**.

Runs entirely on a **free, local model** (Ollama + Qwen2.5-7B). No API keys, no
cost.

```
transcript ─▶ ingest ─▶ 3 specialist agents (∥) ─▶ verifier ─▶ synthesizer ─▶ thesis
                         sentiment · risk · guidance   grounds every quote
```

## Why it's interesting

A 7B open model hallucinates. The core idea here is that a **verifier agent
checks every quote an agent emits against the transcript** (fuzzy substring
match via `rapidfuzz`) and drops anything it can't ground — so the output is
trustworthy despite the small model. Grounding rate is the headline metric.

## Architecture

| Stage | What it does |
|---|---|
| **Ingest** (`wire/ingest.py`) | Parses the transcript into speaker segments with char offsets; splits prepared remarks vs. Q&A; classifies management / analyst / operator. |
| **Specialists** (`wire/agents.py`) | Three agents run in parallel: **sentiment** (confidence vs. hedging, prepared-vs-Q&A shift), **risk** (litigation, softness, charges), **guidance** (raised/lowered/maintained vs. prior). Each has an LLM path and a deterministic heuristic fallback. |
| **Verifier** (`wire/grounding.py` + `pipeline._verify_and_clean`) | Independently re-checks every quote against the transcript, removes unsupported claims, computes a grounding rate, and scales conviction by it. |
| **Synthesizer** | Forms a stance (bullish/neutral/bearish), conviction, headline, catalysts, and a management-credibility read from the *cleaned* findings. |
| **Store / API** (`wire/db.py`, `api.py`) | SQLite persistence; FastAPI service. |
| **UI** (`frontend/`) | Astro single-page app: paste a transcript, watch the thesis render with grounded quotes. |

## Run it

**1. Model (one-time):**
```bash
brew install ollama
ollama serve &                     # if not already running
ollama pull qwen2.5:7b-instruct
```

**2. Backend:**
```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

**3. Frontend (build once):**
```bash
cd frontend && npm install && npm run build && cd ..
```

**4. Go:**
```bash
./venv/bin/python api.py           # → http://localhost:2424  (UI + API)
```

If Ollama isn't running, everything still works in **heuristic mode** (a
deterministic keyword baseline) — the UI badge shows which mode is active.

### CLI
```bash
./venv/bin/python run.py                       # analyze the bundled sample
./venv/bin/python run.py transcript.txt --ticker AAPL --quarter "Q1 2026"
./venv/bin/python run.py --json                # full Thesis as JSON
```

### Eval / tests
```bash
./venv/bin/python benchmark.py 4               # grounding consistency over N runs
./venv/bin/python -m pytest -q                 # unit tests (ingest, grounding, verifier)
```

On the bundled sample the pipeline is deterministic at **100% grounding**.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | model status |
| GET | `/api/sample` | the bundled transcript |
| POST | `/api/analyze` | `{transcript, ticker, quarter}` → Thesis |
| GET | `/api/theses` | stored theses (summaries) |
| GET | `/api/theses/{id}` | one full Thesis |

## Deployment note

The local 7B model is free on your machine but too large for a cheap cloud VM.
Mirroring the **generate-locally / display-in-cloud** split: generation runs on
your Mac and writes theses to SQLite; a small FastAPI + Astro service (Fly.io,
like p4p) serves the stored theses. Deployed live: https://the-wire-hadi.fly.dev

**Cloud generation (also free).** The model layer is provider-agnostic. Set an
OpenAI-compatible endpoint and the deployed site generates in the cloud too —
default is Groq's free tier (Llama-3.3-70B). No key set → the cloud service runs
in heuristic mode; local dev uses Ollama.

```bash
fly secrets set LLM_API_KEY=<groq-key> \
  LLM_BASE_URL=https://api.groq.com/openai/v1 \
  LLM_MODEL=llama-3.3-70b-versatile --app the-wire-hadi
```

| env var | purpose |
|---|---|
| `LLM_API_KEY` | hosted key → selects the OpenAI-compatible client |
| `LLM_BASE_URL` | endpoint (Groq / OpenRouter / HF router / …) |
| `LLM_MODEL` | model id at that provider |
| `WIRE_LLM=heuristic` | force the offline baseline |

## Stack

Python · FastAPI · Ollama (Qwen2.5-7B) · Pydantic · rapidfuzz · SQLite · Astro · pytest

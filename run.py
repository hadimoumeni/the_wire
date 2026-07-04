#!/usr/bin/env python3
"""CLI: analyze an earnings-call transcript and print the investment memo.

    python run.py                         # analyze the bundled sample
    python run.py path/to/transcript.txt --ticker AAPL --quarter "Q1 2026"
    python run.py --json                  # dump the full Thesis as JSON
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from wire.pipeline import run_pipeline
from wire.schemas import Thesis

DEFAULT_TRANSCRIPT = "data/sample_transcript.txt"

_BAR = "─" * 70


def _dot(supported: bool) -> str:
    return "✓" if supported else "✗"


def render(t: Thesis) -> str:
    L = []
    L.append(_BAR)
    L.append(f"  THE WIRE  ·  {t.ticker} · {t.quarter}")
    L.append(f"  {t.headline}")
    L.append(_BAR)
    L.append(f"  STANCE: {t.stance.upper():8}   CONVICTION: {t.conviction}/100"
             f"   [{t.mode} · {t.model_name}]")
    v = t.verification
    L.append(f"  GROUNDING: {v.supported_claims}/{v.total_claims} claims verified "
             f"({v.grounding_rate:.0%})   flagged: {len(v.flagged)}")
    L.append("")
    L.append("  SENTIMENT")
    s = t.sentiment
    L.append(f"    tone={s.tone}  confidence={s.confidence_score}  hedging={s.hedging_score}")
    if s.qa_delta:
        L.append(f"    Q&A shift: {s.qa_delta}")
    for c in s.evidence[:3]:
        L.append(f"      {_dot(c.supported)} \"{c.quote[:80]}\"")
    L.append("")
    L.append("  GUIDANCE")
    for g in t.guidance:
        L.append(f"    • {g.metric}: {g.direction.upper()} — {g.detail[:90]}")
    if not t.guidance:
        L.append("    (none extracted)")
    L.append("")
    L.append("  RISKS")
    for r in t.risks:
        L.append(f"    • [{r.severity}/{r.status}] {r.description[:90]}")
    if not t.risks:
        L.append("    (none extracted)")
    L.append("")
    L.append("  CATALYSTS: " + (", ".join(t.catalysts) or "—"))
    L.append(f"  MGMT CREDIBILITY: {t.mgmt_credibility}")
    if v.flagged:
        L.append("")
        L.append("  ⚠ VERIFIER FLAGGED (removed, not grounded):")
        for c in v.flagged[:5]:
            L.append(f"      ✗ ({c.score:.0f}) \"{c.quote[:70]}\"")
    L.append(_BAR)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="The Wire — earnings-call analyzer")
    ap.add_argument("transcript", nargs="?", default=DEFAULT_TRANSCRIPT)
    ap.add_argument("--ticker", default="NWL")
    ap.add_argument("--quarter", default="Q3 2026")
    ap.add_argument("--json", action="store_true", help="print full Thesis JSON")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    try:
        transcript = open(args.transcript, encoding="utf-8").read()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    thesis = asyncio.run(run_pipeline(
        transcript, ticker=args.ticker, quarter=args.quarter,
        persist=not args.no_persist))

    if args.json:
        print(thesis.model_dump_json(indent=2))
    else:
        print(render(thesis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

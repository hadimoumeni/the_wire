#!/usr/bin/env python3
"""Eval harness — runs the pipeline N times on the sample and reports how well
the model's quotes ground against the transcript. This is the metric that
matters for a small open model: high grounding_rate = trustworthy output.

    python benchmark.py            # 3 runs on the bundled sample
    python benchmark.py 5 AAPL.txt
"""
from __future__ import annotations

import asyncio
import statistics
import sys

from wire.pipeline import run_pipeline


async def main(n: int, path: str) -> None:
    transcript = open(path, encoding="utf-8").read()
    rates, guids, flags = [], [], []
    print(f"Benchmarking {path} — {n} runs\n")
    print(f"{'run':>3}  {'stance':8} {'conv':>4} {'ground':>7} {'claims':>7} "
          f"{'guid':>4} {'risk':>4} {'flag':>4}")
    for i in range(1, n + 1):
        t = await run_pipeline(transcript, ticker="NWL", quarter="Q3 2026", persist=False)
        v = t.verification
        rates.append(v.grounding_rate)
        guids.append(len(t.guidance))
        flags.append(len(v.flagged))
        print(f"{i:>3}  {t.stance:8} {t.conviction:>4} "
              f"{v.grounding_rate:>7.0%} {v.supported_claims:>3}/{v.total_claims:<3} "
              f"{len(t.guidance):>4} {len(t.risks):>4} {len(v.flagged):>4}")

    print("\nsummary:")
    print(f"  grounding_rate  mean={statistics.mean(rates):.0%}  "
          f"min={min(rates):.0%}  max={max(rates):.0%}")
    print(f"  guidance items  mean={statistics.mean(guids):.1f}")
    print(f"  flagged quotes  mean={statistics.mean(flags):.1f}")
    ok = min(rates) >= 0.7 and statistics.mean(guids) >= 1
    print(f"\n  {'PASS' if ok else 'WEAK'}: grounding {'reliably high' if ok else 'inconsistent'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_transcript.txt"
    asyncio.run(main(n, path))

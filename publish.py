#!/usr/bin/env python3
"""Publish locally-generated theses to the deployed (display-only) site.

The generate-locally / display-in-cloud split: you run the real model on your
Mac (free), then publish the results to the live site. The cloud service seeds
its DB from seed/theses.json on boot, so this just refreshes that seed from your
local DB. Only real-model (Ollama) theses are published by default — the cloud
should show real output, not the heuristic baseline.

    python run.py data/aapl.txt --ticker AAPL --quarter "Q1 2026"  # generate
    python publish.py --deploy                                     # publish live
"""
from __future__ import annotations

import argparse
import json
import subprocess

from wire import db

SEED = "seed/theses.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish local theses to the live site")
    ap.add_argument("--latest", type=int, default=0, help="only the N most recent (0 = all)")
    ap.add_argument("--include-heuristic", action="store_true",
                    help="also publish heuristic-mode theses (default: real-model only)")
    ap.add_argument("--deploy", action="store_true", help="run `fly deploy` afterwards")
    args = ap.parse_args()

    rows = db.list_theses()
    if not args.include_heuristic:
        rows = [r for r in rows if r["mode"] == "ollama"]
    if args.latest:
        rows = rows[: args.latest]

    full = [db.get_thesis(r["id"]).model_dump() for r in rows]
    with open(SEED, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    print(f"wrote {len(full)} thesis(es) → {SEED}")
    for t in full:
        v = t["verification"]
        print(f"  · {t['ticker']} {t['quarter']}  {t['stance']} conv={t['conviction']} "
              f"grounding={v['grounding_rate']:.0%} ({t['mode']})")

    if not full:
        print("\n(nothing to publish — generate a real-model thesis first with run.py)")
        return

    if args.deploy:
        print("\ndeploying…")
        subprocess.run(["fly", "deploy", "--ha=false"], check=True)
    else:
        print("\nnext:  git commit -am 'update theses' && fly deploy   (or rerun with --deploy)")


if __name__ == "__main__":
    main()

"""Orchestration: transcript in → verified Thesis out.

Flow:
    1. parse transcript into speaker segments
    2. run the 3 specialist agents concurrently (sentiment / risk / guidance)
    3. VERIFIER re-checks every quote against the transcript, drops unsupported
       claims, and computes a grounding rate
    4. synthesizer forms a stance from the *cleaned* findings
    5. conviction is scaled by the grounding rate; thesis is persisted
"""
from __future__ import annotations

import asyncio
import datetime

from . import agents, db
from .grounding import check_quote
from .ingest import parse_transcript, management_text
from .llm import OllamaClient, get_client
from .schemas import (
    Citation, GuidanceItem, RiskItem, SentimentReport, Thesis, VerifyReport,
)


def _verify_and_clean(
    sentiment: SentimentReport,
    risks: list[RiskItem],
    guidance: list[GuidanceItem],
    transcript: str,
):
    """Independently re-check every quote; keep only grounded claims.

    Returns (clean_sentiment, clean_risks, clean_guidance, VerifyReport).
    Items lacking a supported quote are flagged and removed — no claim without
    verifiable evidence.
    """
    total = 0
    supported = 0
    flagged: list[Citation] = []

    # Sentiment evidence
    clean_ev: list[Citation] = []
    for c in sentiment.evidence:
        total += 1
        v = check_quote(c.quote, transcript)
        v.claim = c.claim
        if v.supported:
            supported += 1
            clean_ev.append(v)
        else:
            flagged.append(v)
    sentiment = sentiment.model_copy(update={"evidence": clean_ev})

    def _clean_items(items, label):
        nonlocal total, supported
        kept = []
        for it in items:
            total += 1
            if it.citation and it.citation.quote.strip():
                v = check_quote(it.citation.quote, transcript)
                v.claim = getattr(it, "description", None) or getattr(it, "metric", None)
                it = it.model_copy(update={"citation": v})
                if v.supported:
                    supported += 1
                    kept.append(it)
                else:
                    flagged.append(v)
            else:
                flagged.append(Citation(quote="(no quote provided)",
                                        claim=f"{label}: unverifiable claim",
                                        supported=False))
        return kept

    risks = _clean_items(risks, "risk")
    guidance = _clean_items(guidance, "guidance")

    rate = round(supported / total, 3) if total else 0.0
    report = VerifyReport(
        grounding_rate=rate, total_claims=total, supported_claims=supported,
        flagged=flagged,
        notes=(f"{supported}/{total} claims grounded in the transcript; "
               f"{len(flagged)} unsupported claim(s) removed."),
    )
    return sentiment, risks, guidance, report


async def run_pipeline(transcript: str, ticker: str = "", quarter: str = "",
                       llm: OllamaClient | None = None, persist: bool = True) -> Thesis:
    if llm is None:
        llm = get_client()
    mode = llm.provider if llm is not None else "heuristic"
    model_name = llm.model if llm is not None else "heuristic-baseline"

    segments = parse_transcript(transcript)
    prepared = management_text(segments, section="prepared_remarks")
    qa = management_text(segments, section="qa")
    mgmt_all = management_text(segments)

    # 2 + 3. specialists (concurrent) → verifier, with a self-correction retry:
    # if the model fabricated everything (nothing grounds), try once more and
    # keep the better attempt. Deterministic heuristic mode never retries.
    attempts = 2 if llm is not None else 1
    best = None
    for i in range(attempts):
        s, r, g = await asyncio.gather(
            asyncio.to_thread(agents.sentiment_agent, prepared, qa, transcript, llm),
            asyncio.to_thread(agents.risk_agent, mgmt_all, transcript, llm),
            asyncio.to_thread(agents.guidance_agent, mgmt_all, transcript, llm),
        )
        cs, cr, cg, report = _verify_and_clean(s, r, g, transcript)
        if best is None or report.supported_claims > best[3].supported_claims:
            best = (cs, cr, cg, report)
        if report.supported_claims > 0 and report.grounding_rate >= 0.5:
            break
        if i + 1 < attempts:
            report.notes += " Low grounding — retried once."
    sentiment, risks, guidance, report = best

    # 4. synthesize from cleaned findings
    synth = await asyncio.to_thread(agents.synthesize, sentiment, risks, guidance, llm)

    # 5. conviction scaled by grounding
    conviction = round(synth.conviction * report.grounding_rate)

    thesis = Thesis(
        ticker=ticker or "N/A", quarter=quarter or "N/A",
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        model_name=model_name, mode=mode,
        stance=synth.stance, conviction=conviction, headline=synth.headline,
        sentiment=sentiment, risks=risks, guidance=guidance,
        catalysts=synth.catalysts, mgmt_credibility=synth.mgmt_credibility,
        verification=report,
    )

    if persist:
        thesis.id = db.save_thesis(thesis)
    return thesis

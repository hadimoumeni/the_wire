"""The four agents.

Each specialist has two implementations:
  * an LLM path (Ollama, structured output), used when a model is available;
  * a deterministic heuristic path, used as a fallback so the pipeline always
    runs (and so tests are fast and offline).

Every quote either path produces is grounded against the transcript via
`grounding.check_quote`, so a fabricated quote is caught regardless of source.
"""
from __future__ import annotations

import re
from pydantic import ValidationError

from . import prompts
from .grounding import check_quote
from .llm import OllamaClient, OllamaError
from .schemas import (
    Citation, SentimentReport, RiskItem, GuidanceItem, SynthesisLLMOut,
    SentimentLLMOut, RiskLLMOut, GuidanceLLMOut,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("["):
            continue
        out.extend(s.strip() for s in _SENT_SPLIT.split(line) if len(s.strip()) > 12)
    return out


def _cite(quote: str, transcript: str, claim: str | None = None) -> Citation:
    c = check_quote(quote, transcript)
    c.claim = claim
    return c


def _has(s: str, *terms: str) -> bool:
    low = s.lower()
    return any(t in low for t in terms)


# =========================================================================== #
# Sentiment
# =========================================================================== #
_POS = ("strong", "confident", "pleased", "healthy", "executing", "paying off",
        "expanded", "expansion", "record", "momentum", "solid", "terrific",
        "committed", "well", "improve", "sustainable")
_HEDGE = ("uncertain", "cautious", "challenging", "difficult to say", "don't want",
          "do not want", "choppy", "softness", "pressure", "speculate", "monitor",
          "bumpy", "cannot predict", "prudent", "conservativ", "um,", "honestly",
          "get ahead of", "we'll know more", "under pressure")


def sentiment_agent(prepared: str, qa: str, transcript: str,
                    llm: OllamaClient | None) -> SentimentReport:
    if llm is not None:
        try:
            raw = llm.chat_json(prompts.SENTIMENT_SYS,
                                prompts.sentiment_user(prepared, qa),
                                prompts.SENTIMENT_SCHEMA)
            out = SentimentLLMOut.model_validate(raw)
            ev = [_cite(e.quote, transcript, e.claim) for e in out.evidence if e.quote.strip()]
            return SentimentReport(
                tone=out.tone, confidence_score=out.confidence_score,
                hedging_score=out.hedging_score, prepared_tone=out.prepared_tone,
                qa_tone=out.qa_tone, qa_delta=out.qa_delta, summary=out.summary,
                evidence=ev,
            )
        except (OllamaError, ValidationError):
            pass
    return _sentiment_heuristic(prepared, qa, transcript)


def _sentiment_heuristic(prepared: str, qa: str, transcript: str) -> SentimentReport:
    all_sents = _sentences(prepared) + _sentences(qa)
    pos_hits = sum(1 for s in all_sents if _has(s, *_POS))
    hedge_hits = sum(1 for s in all_sents if _has(s, *_HEDGE))
    confidence = max(0, min(100, 30 + pos_hits * 8))
    hedging = max(0, min(100, 15 + hedge_hits * 10))
    if hedging > confidence + 10:
        tone = "bearish"
    elif confidence > hedging + 10:
        tone = "bullish"
    else:
        tone = "neutral"

    prep_hedge = sum(1 for s in _sentences(prepared) if _has(s, *_HEDGE))
    qa_hedge = sum(1 for s in _sentences(qa) if _has(s, *_HEDGE))
    if qa_hedge > prep_hedge:
        delta = ("Management is noticeably more hedged in the unscripted Q&A "
                 "than in prepared remarks — a caution signal.")
    else:
        delta = "Tone is broadly consistent between prepared remarks and Q&A."

    scored = sorted(all_sents,
                    key=lambda s: sum(t in s.lower() for t in _POS + _HEDGE),
                    reverse=True)
    evidence: list[Citation] = []
    for s in scored:
        if len(evidence) >= 4:
            break
        c = _cite(s, transcript, claim="tone signal")
        if c.supported:
            evidence.append(c)

    return SentimentReport(
        tone=tone, confidence_score=confidence, hedging_score=hedging,
        prepared_tone="confident, on-message" if tone != "bearish" else "measured",
        qa_tone="hedged" if qa_hedge >= prep_hedge else "steady",
        qa_delta=delta,
        summary=(f"Heuristic read: {pos_hits} confidence cues vs {hedge_hits} hedging "
                 f"cues → {tone} tone. {delta}"),
        evidence=evidence,
    )


# =========================================================================== #
# Risk
# =========================================================================== #
_RISK_HIGH = ("litigation", "dispute", "cannot predict", "lawsuit", "charge",
              "investigation", "impairment")
_RISK_MED = ("softness", "pressure", "choppy", "headwind", "challenging",
             "uncertainty", "below our prior", "cautious", "slowdown", "weakness",
             "pull-forward", "declin")


def risk_agent(mgmt_text: str, transcript: str,
               llm: OllamaClient | None) -> list[RiskItem]:
    if llm is not None:
        try:
            raw = llm.chat_json(prompts.RISK_SYS, prompts.risk_user(mgmt_text),
                                prompts.RISK_SCHEMA)
            out = RiskLLMOut.model_validate(raw)
            items = []
            for r in out.risks:
                items.append(RiskItem(
                    description=r.description, severity=r.severity, status=r.status,
                    citation=_cite(r.quote, transcript) if r.quote.strip() else None,
                ))
            if items:
                return items
        except (OllamaError, ValidationError):
            pass
    return _risk_heuristic(mgmt_text, transcript)


def _risk_heuristic(mgmt_text: str, transcript: str) -> list[RiskItem]:
    items: list[RiskItem] = []
    seen = set()
    for s in _sentences(mgmt_text):
        high = _has(s, *_RISK_HIGH)
        med = _has(s, *_RISK_MED)
        if not (high or med):
            continue
        key = s[:40].lower()
        if key in seen:
            continue
        seen.add(key)
        cite = _cite(s, transcript)
        if not cite.supported:
            continue
        items.append(RiskItem(
            description=s if len(s) < 160 else s[:157] + "...",
            severity="high" if high else "medium",
            status="escalating" if _has(s, "cannot predict", "persisted", "longer than") else "recurring",
            citation=cite,
        ))
        if len(items) >= 6:
            break
    return items


# =========================================================================== #
# Guidance
# =========================================================================== #
def _guidance_metric(s: str) -> str:
    low = s.lower()
    if "eps" in low or "earnings per share" in low:
        return "Adjusted EPS"
    if "margin" in low:
        return "Gross margin"
    if "revenue" in low or "billion" in low or "million" in low:
        return "Revenue"
    return "Outlook"


def _guidance_direction(s: str) -> str:
    low = s.lower()
    if _has(low, "below our prior", "lowering", "trimming", "reduc", "more cautious", "cut"):
        return "lowered"
    if _has(low, "raising", "above the high end", "ahead of", "increas"):
        return "raised"
    if _has(low, "no change", "maintain", "reaffirm", "unchanged"):
        return "maintained"
    if _has(low, "withdraw", "suspend"):
        return "withdrawn"
    if _has(low, "we now expect", "we expect", "for the fourth quarter", "full-year", "outlook"):
        return "introduced"
    return "unclear"


def guidance_agent(mgmt_text: str, transcript: str,
                   llm: OllamaClient | None) -> list[GuidanceItem]:
    if llm is not None:
        try:
            raw = llm.chat_json(prompts.GUIDANCE_SYS, prompts.guidance_user(mgmt_text),
                                prompts.GUIDANCE_SCHEMA)
            out = GuidanceLLMOut.model_validate(raw)
            items = []
            for g in out.guidance:
                items.append(GuidanceItem(
                    metric=g.metric, direction=g.direction, detail=g.detail,
                    citation=_cite(g.quote, transcript) if g.quote.strip() else None,
                ))
            if items:
                return items
        except (OllamaError, ValidationError):
            pass
    return _guidance_heuristic(mgmt_text, transcript)


_GUID_CUES = ("guidance", "we now expect", "we expect", "full-year", "outlook",
              "range of", "for the fourth quarter", "adjusted eps", "prior guidance",
              "below our prior", "lowering", "trimming", "raising")


def _guidance_heuristic(mgmt_text: str, transcript: str) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    seen = set()
    for s in _sentences(mgmt_text):
        if not _has(s, *_GUID_CUES):
            continue
        key = s[:40].lower()
        if key in seen:
            continue
        seen.add(key)
        cite = _cite(s, transcript)
        if not cite.supported:
            continue
        items.append(GuidanceItem(
            metric=_guidance_metric(s), direction=_guidance_direction(s),
            detail=s if len(s) < 200 else s[:197] + "...", citation=cite,
        ))
        if len(items) >= 6:
            break
    return items


# =========================================================================== #
# Synthesizer
# =========================================================================== #
def synthesize(sentiment: SentimentReport, risks: list[RiskItem],
               guidance: list[GuidanceItem], llm: OllamaClient | None) -> SynthesisLLMOut:
    if llm is not None:
        try:
            raw = llm.chat_json(
                prompts.SYNTHESIS_SYS,
                prompts.synthesis_user(
                    sentiment.model_dump_json(),
                    RiskLLMOut(risks=[]).model_dump_json() if not risks else
                    "[" + ",".join(r.model_dump_json() for r in risks) + "]",
                    "[" + ",".join(g.model_dump_json() for g in guidance) + "]",
                ),
                prompts.SYNTHESIS_SCHEMA,
            )
            return SynthesisLLMOut.model_validate(raw)
        except (OllamaError, ValidationError):
            pass
    return _synth_heuristic(sentiment, risks, guidance)


def _synth_heuristic(sentiment: SentimentReport, risks: list[RiskItem],
                     guidance: list[GuidanceItem]) -> SynthesisLLMOut:
    lowered = [g for g in guidance if g.direction in ("lowered", "withdrawn")]
    raised = [g for g in guidance if g.direction == "raised"]
    high_risks = [r for r in risks if r.severity == "high"]

    if lowered and not raised:
        stance = "bearish"
    elif raised and not lowered:
        stance = "bullish"
    else:
        stance = sentiment.tone

    conviction = 55
    if lowered:
        conviction += 10
    if sentiment.hedging_score > 60:
        conviction += 5

    parts = []
    if lowered:
        parts.append("guidance cut")
    if high_risks:
        parts.append(f"{len(high_risks)} high-severity risk(s)")
    if sentiment.hedging_score > 55:
        parts.append("elevated Q&A hedging")
    headline = ("Northbound signals mixed: " + ", ".join(parts)) if parts else \
        "Steady quarter; tone constructive"

    catalysts = []
    for g in guidance:
        if g.direction == "raised":
            catalysts.append(f"{g.metric} guidance raised")
    if any("margin" in (g.metric or "").lower() for g in guidance):
        catalysts.append("Margin trajectory")
    if not catalysts:
        catalysts = ["Next-quarter guidance update"]

    cred = ("Management hedged and deflected specifics in Q&A"
            if sentiment.hedging_score > 55 else
            "Management answers were direct and consistent")

    return SynthesisLLMOut(stance=stance, conviction=min(conviction, 100),
                           headline=headline, catalysts=catalysts[:4],
                           mgmt_credibility=cred)

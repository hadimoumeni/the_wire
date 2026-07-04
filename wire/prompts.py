"""Prompts and JSON schemas for each specialist agent.

Schemas are hand-written (no $ref) so Ollama's structured-output/GBNF path
compiles them cleanly for a 7B model. Prompts hammer one rule: quotes must be
copied verbatim from the transcript — that's what lets the verifier catch
fabrications.
"""
from __future__ import annotations

VERBATIM_RULE = (
    "CRITICAL QUOTING RULE: every `quote` value MUST be an exact, character-for-"
    "character substring copied from the transcript text provided below. Copy a "
    "whole sentence verbatim — do NOT paraphrase, summarize, shorten, fix grammar, "
    "or write your own words. A quote that is not an exact copy will be discarded "
    "by a downstream verifier and your finding will be lost, so only use real "
    "sentences from the text. If no exact sentence supports a point, omit that "
    "point. Never invent numbers, names, or quotes."
)

# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "tone": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "confidence_score": {"type": "integer"},
        "hedging_score": {"type": "integer"},
        "prepared_tone": {"type": "string"},
        "qa_tone": {"type": "string"},
        "qa_delta": {"type": "string"},
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["claim", "quote"],
            },
        },
    },
    "required": ["tone", "confidence_score", "hedging_score", "summary", "evidence"],
}

SENTIMENT_SYS = (
    "You are a sell-side analyst reading management's tone on an earnings call. "
    "Assess how confident vs. hedging management sounds, and how the tone shifts "
    "between scripted prepared remarks and the unscripted Q&A (where evasiveness "
    "shows). Scores are 0-100. " + VERBATIM_RULE
)

def sentiment_user(prepared: str, qa: str) -> str:
    return (
        "PREPARED REMARKS (management):\n" + (prepared or "(none)") +
        "\n\nQ&A ANSWERS (management):\n" + (qa or "(none)") +
        "\n\nReturn the sentiment JSON. Give 2-4 evidence items with verbatim quotes, "
        "preferring quotes that reveal hedging or confidence."
    )

# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "status": {"type": "string", "enum": ["new", "recurring", "escalating"]},
                    "quote": {"type": "string"},
                },
                "required": ["description", "severity", "status", "quote"],
            },
        }
    },
    "required": ["risks"],
}

RISK_SYS = (
    "You are a risk analyst. Extract the key risks management disclosed or "
    "revealed on the call: litigation, demand softness, macro headwinds, "
    "guidance uncertainty, one-time charges. Rate severity and whether each risk "
    "is new, recurring, or escalating. " + VERBATIM_RULE
)

def risk_user(mgmt_text: str) -> str:
    return (
        "MANAGEMENT REMARKS AND ANSWERS:\n" + (mgmt_text or "(none)") +
        "\n\nReturn the risks JSON with 2-6 risks, each backed by a verbatim quote."
    )

# --------------------------------------------------------------------------- #
# Guidance
# --------------------------------------------------------------------------- #
GUIDANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "guidance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["raised", "lowered", "maintained", "introduced", "withdrawn", "unclear"],
                    },
                    "detail": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["metric", "direction", "detail", "quote"],
            },
        }
    },
    "required": ["guidance"],
}

GUIDANCE_SYS = (
    "You extract forward guidance from an earnings call. For each guided metric "
    "(revenue, EPS, margins, segment outlook), state the metric, whether it was "
    "raised/lowered/maintained/introduced/withdrawn versus the prior outlook, and "
    "the specifics. " + VERBATIM_RULE
)

def guidance_user(mgmt_text: str) -> str:
    return (
        "MANAGEMENT REMARKS AND ANSWERS:\n" + (mgmt_text or "(none)") +
        "\n\nReturn the guidance JSON. Capture every explicit forward-looking number "
        "or outlook, each with a verbatim quote."
    )

# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "conviction": {"type": "integer"},
        "headline": {"type": "string"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "mgmt_credibility": {"type": "string"},
    },
    "required": ["stance", "conviction", "headline", "catalysts", "mgmt_credibility"],
}

SYNTHESIS_SYS = (
    "You are a portfolio manager writing a one-paragraph investment read from "
    "three analysts' findings (sentiment, risk, guidance). Take a stance "
    "(bullish/neutral/bearish) with a conviction score 0-100, a punchy one-line "
    "headline, near-term catalysts, and a candid read on management credibility "
    "(did they dodge questions? hedge?). Base everything ONLY on the findings "
    "provided; do not invent facts."
)

def synthesis_user(sentiment_json: str, risks_json: str, guidance_json: str) -> str:
    return (
        "SENTIMENT FINDINGS:\n" + sentiment_json +
        "\n\nRISK FINDINGS:\n" + risks_json +
        "\n\nGUIDANCE FINDINGS:\n" + guidance_json +
        "\n\nReturn the synthesis JSON."
    )

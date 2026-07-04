"""Pydantic data model for The Wire.

Two layers:
  * Rich domain models (Segment, Thesis, ...) — what the pipeline assembles and
    the API/UI consume.
  * Thin per-agent LLM output models (*_LLMOut) — the exact JSON shape each agent
    asks the model for. Kept small so a 7B model reliably fills them.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

Section = Literal["prepared_remarks", "qa"]
RoleType = Literal["management", "analyst", "operator", "unknown"]
Stance = Literal["bullish", "neutral", "bearish"]
Severity = Literal["low", "medium", "high"]
RiskStatus = Literal["new", "recurring", "escalating"]
GuidanceDirection = Literal["raised", "lowered", "maintained", "introduced", "withdrawn", "unclear"]


# ---------------------------------------------------------------------------
# Transcript structure
# ---------------------------------------------------------------------------
class Segment(BaseModel):
    id: int
    speaker: str
    role: str = ""
    role_type: RoleType = "unknown"
    section: Section = "prepared_remarks"
    text: str
    start: int  # char offset into the raw transcript
    end: int


class Citation(BaseModel):
    """A claimed quote, checked against the transcript by the verifier."""
    quote: str
    claim: Optional[str] = None  # the agent's interpretation this quote supports
    supported: bool = False
    score: float = 0.0          # 0-100 fuzzy-match confidence
    start: Optional[int] = None  # char span located in the transcript
    end: Optional[int] = None
    matched_text: Optional[str] = None  # what the transcript actually says there


# ---------------------------------------------------------------------------
# Specialist outputs (rich)
# ---------------------------------------------------------------------------
class SentimentReport(BaseModel):
    tone: Stance = "neutral"
    confidence_score: int = 50   # 0-100, how confident/upbeat management sounds
    hedging_score: int = 50      # 0-100, how much they hedge/qualify
    prepared_tone: str = ""
    qa_tone: str = ""
    qa_delta: str = ""           # how tone shifts from scripted remarks to live Q&A
    summary: str = ""
    evidence: list[Citation] = Field(default_factory=list)


class RiskItem(BaseModel):
    description: str
    severity: Severity = "medium"
    status: RiskStatus = "recurring"
    citation: Optional[Citation] = None


class GuidanceItem(BaseModel):
    metric: str                  # e.g. "Revenue", "EPS", "Gross margin"
    direction: GuidanceDirection = "unclear"
    detail: str = ""
    citation: Optional[Citation] = None


class VerifyReport(BaseModel):
    grounding_rate: float = 0.0  # fraction of claims whose quotes were found
    total_claims: int = 0
    supported_claims: int = 0
    flagged: list[Citation] = Field(default_factory=list)  # unsupported quotes
    notes: str = ""


class Thesis(BaseModel):
    id: Optional[int] = None
    ticker: str
    quarter: str
    created_at: str = ""
    model_name: str = ""
    mode: str = "ollama"         # "ollama" | "heuristic"

    stance: Stance = "neutral"
    conviction: int = 0          # 0-100, scaled down by grounding rate
    headline: str = ""

    sentiment: SentimentReport = Field(default_factory=SentimentReport)
    risks: list[RiskItem] = Field(default_factory=list)
    guidance: list[GuidanceItem] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    mgmt_credibility: str = ""

    verification: VerifyReport = Field(default_factory=VerifyReport)


# ---------------------------------------------------------------------------
# Thin LLM output schemas (what each agent asks the model to return)
# ---------------------------------------------------------------------------
class _Quoted(BaseModel):
    claim: str
    quote: str = ""


class SentimentLLMOut(BaseModel):
    tone: Stance = "neutral"
    confidence_score: int = 50
    hedging_score: int = 50
    prepared_tone: str = ""
    qa_tone: str = ""
    qa_delta: str = ""
    summary: str = ""
    evidence: list[_Quoted] = Field(default_factory=list)


class _RiskLLM(BaseModel):
    description: str
    severity: Severity = "medium"
    status: RiskStatus = "recurring"
    quote: str = ""


class RiskLLMOut(BaseModel):
    risks: list[_RiskLLM] = Field(default_factory=list)


class _GuidanceLLM(BaseModel):
    metric: str
    direction: GuidanceDirection = "unclear"
    detail: str = ""
    quote: str = ""


class GuidanceLLMOut(BaseModel):
    guidance: list[_GuidanceLLM] = Field(default_factory=list)


class SynthesisLLMOut(BaseModel):
    stance: Stance = "neutral"
    conviction: int = 50
    headline: str = ""
    catalysts: list[str] = Field(default_factory=list)
    mgmt_credibility: str = ""

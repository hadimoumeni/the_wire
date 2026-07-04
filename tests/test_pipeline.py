import asyncio
import os

os.environ["WIRE_LLM"] = "heuristic"  # deterministic, offline

from wire.pipeline import run_pipeline, _verify_and_clean
from wire.schemas import SentimentReport, Citation, RiskItem, GuidanceItem

RAW = open("data/sample_transcript.txt", encoding="utf-8").read()


def test_verifier_flags_and_removes_fabrication():
    sentiment = SentimentReport(evidence=[
        Citation(quote="Pricing has been under pressure and volumes have been choppy"),
        Citation(quote="We are raising guidance to five billion dollars"),  # fabricated
    ])
    risks = [RiskItem(description="carrier dispute",
                      citation=Citation(quote="we recorded a $22 million charge related to an ongoing dispute"))]
    guidance = [GuidanceItem(metric="Revenue", direction="lowered",
                             citation=Citation(quote="revenue of nine hundred trillion dollars"))]  # fabricated

    s, r, g, report = _verify_and_clean(sentiment, risks, guidance, RAW)

    # Fabricated sentiment quote removed; real one kept.
    kept = [c.quote for c in s.evidence]
    assert any("choppy" in q for q in kept)
    assert not any("five billion" in q for q in kept)
    # Fabricated guidance dropped, real risk kept.
    assert len(g) == 0
    assert len(r) == 1
    # Two fabrications flagged, grounding rate below 1.
    assert report.total_claims == 4
    assert report.supported_claims == 2
    assert 0.0 < report.grounding_rate < 1.0
    assert len(report.flagged) == 2


def test_format_agnostic_unstructured_text():
    # No speaker labels, no structure — a raw paragraph. The parser finds no
    # speakers, so the pipeline must fall back to analyzing the raw transcript.
    from wire.ingest import parse_transcript, stats
    text = (
        "Northwind reported third quarter revenue of $842 million, up 11 percent. "
        "The company is lowering its full-year revenue guidance to a range of "
        "$3.28 billion to $3.31 billion, below its prior guidance, citing continued "
        "softness in spot-market freight volumes. Management also disclosed a "
        "$22 million charge related to an ongoing dispute with a regional carrier "
        "partner, noting that litigation of this nature carries inherent uncertainty."
    )
    assert stats(parse_transcript(text))["management_turns"] == 0  # no speakers found
    thesis = asyncio.run(run_pipeline(text, persist=False))
    assert thesis.guidance or thesis.risks, "should extract from raw text"
    assert thesis.verification.total_claims > 0
    assert all(c.supported for c in thesis.sentiment.evidence)


def test_end_to_end_heuristic():
    thesis = asyncio.run(run_pipeline(RAW, ticker="NWL", quarter="Q3 2026", persist=False))
    assert thesis.stance in ("bullish", "neutral", "bearish")
    assert thesis.guidance, "should extract some guidance"
    # Every surviving evidence quote must be grounded.
    assert all(c.supported for c in thesis.sentiment.evidence)
    # Guidance cut should be detected somewhere.
    assert any(g.direction == "lowered" for g in thesis.guidance)
    assert thesis.verification.total_claims > 0

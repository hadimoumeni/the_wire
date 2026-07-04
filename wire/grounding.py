"""Quote grounding: does a claimed quote actually appear in the transcript?

This is what makes a small, hallucination-prone open model usable for finance.
Every quote an agent emits is checked here with fuzzy substring matching
(``rapidfuzz.partial_ratio``), which tolerates whitespace/paraphrase noise while
still catching invented quotes. Only lowercasing is applied to the haystack, so
the returned character span maps back to the raw transcript.
"""
from __future__ import annotations

import re
from rapidfuzz import fuzz
from .schemas import Citation

# A quote scoring at/above this is treated as genuinely present in the source.
SUPPORT_THRESHOLD = 82.0
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def check_quote(quote: str, transcript: str) -> Citation:
    """Locate `quote` in `transcript` and return a Citation with the verdict."""
    q = _norm(quote)
    if len(q) < 8:
        # Too short to verify meaningfully — treat as unsupported.
        return Citation(quote=quote, supported=False, score=0.0)

    hay = transcript.lower()
    ndl = q.lower()
    try:
        al = fuzz.partial_ratio_alignment(ndl, hay)
    except Exception:
        al = None

    if al is None:
        score = fuzz.partial_ratio(ndl, hay)
        supported = score >= SUPPORT_THRESHOLD
        return Citation(quote=quote, supported=supported, score=round(score, 1))

    score = al.score
    supported = score >= SUPPORT_THRESHOLD
    start, end = al.dest_start, al.dest_end
    matched = transcript[start:end] if supported else None
    return Citation(
        quote=quote,
        supported=supported,
        score=round(score, 1),
        start=start if supported else None,
        end=end if supported else None,
        matched_text=matched,
    )


def verify_all(citations: list[Citation], transcript: str) -> list[Citation]:
    """Re-check a list of citations against the transcript, in place-ish."""
    out = []
    for c in citations:
        out.append(check_quote(c.quote, transcript))
    return out

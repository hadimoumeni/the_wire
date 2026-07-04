"""Parse an earnings-call transcript into structured, offset-tracked segments.

Handles the common "Motley Fool" layout:

    Operator

    Good day, and welcome to ...

    Tim Cook -- Chief Executive Officer

    Thank you. ...

    Questions & Answers:

    Operator

    [Analyst question] ...

Speaker headers are lines like ``Name -- Role`` or ``Name, Role`` or ``NAME:``.
Everything between two headers is that speaker's turn. Char offsets into the raw
transcript are preserved so the verifier can point back to exact source spans.
"""
from __future__ import annotations

import re
from .schemas import Segment, RoleType

# Q&A section boundary markers (case-insensitive).
_QA_MARKERS = [
    r"questions?\s*&\s*answers?",
    r"questions?\s+and\s+answers?",
    r"question[- ]and[- ]answer\s+session",
    r"q\s*&\s*a\s+session",
]

# Speaker header patterns. Group 1 = name, group 2 = role (optional).
_HEADER_PATTERNS = [
    re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}?)\s+--\s+(.{2,60})$"),   # Name -- Role
    re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}?),\s+(.{2,60})$"),       # Name, Role
]
_OPERATOR_RE = re.compile(r"^\s*operator\s*$", re.IGNORECASE)
_COLON_HEADER_RE = re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}):\s*$")    # NAME:

_MGMT_ROLE_HINTS = (
    "chief", "ceo", "cfo", "coo", "cto", "president", "founder", "officer",
    "vp", "vice president", "head of", "director of", "treasurer", "controller",
    "investor relations", "ir ",
)
_ANALYST_ROLE_HINTS = ("analyst", "research", "securities", "capital", "partners",
                        "bank", "& co", "llc", "equity")


def _find_qa_boundary(text: str) -> int:
    """Return the char offset where the Q&A section begins, or -1.

    Matches only a *standalone* Q&A header line (e.g. ``Questions & Answers:``),
    not an inline mention like "there will be a question-and-answer session" in
    the operator's intro.
    """
    line_re = re.compile(r"^\s*(?:%s)\s*:?\s*$" % "|".join(_QA_MARKERS), re.IGNORECASE)
    offset = 0
    for line in text.split("\n"):
        if line_re.match(line):
            return offset
        offset += len(line) + 1
    return -1


def _classify_role(role: str) -> RoleType:
    r = role.lower()
    if any(h in r for h in _MGMT_ROLE_HINTS):
        return "management"
    if any(h in r for h in _ANALYST_ROLE_HINTS):
        return "analyst"
    return "unknown"


def _match_header(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if _OPERATOR_RE.match(s):
        return ("Operator", "Operator")
    # Sentence lines end in terminal punctuation — never a speaker header.
    if s[-1] in ".?!":
        return None
    for pat in _HEADER_PATTERNS:
        m = pat.match(s)
        if m:
            role = m.group(2).strip()
            if role.count(",") >= 1:   # a comma in the "role" means it's prose
                return None
            return (m.group(1).strip(), role)
    m = _COLON_HEADER_RE.match(s)
    if m:
        return (m.group(1).strip(), "")
    return None


def parse_transcript(raw: str) -> list[Segment]:
    """Split a transcript into speaker segments with char offsets."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    qa_start = _find_qa_boundary(text)

    lines = text.split("\n")
    segments: list[Segment] = []
    cursor = 0                 # running char offset into `text`
    cur_speaker: str | None = None
    cur_role = ""
    cur_start = 0
    buf: list[str] = []
    seg_id = 0

    def flush(end_offset: int) -> None:
        nonlocal seg_id
        if cur_speaker is None:
            return
        body = "\n".join(buf).strip()
        if not body:
            return
        role_type = "operator" if cur_speaker == "Operator" else _classify_role(cur_role)
        section = "qa" if (qa_start != -1 and cur_start >= qa_start) else "prepared_remarks"
        segments.append(Segment(
            id=seg_id, speaker=cur_speaker, role=cur_role, role_type=role_type,
            section=section, text=body, start=cur_start, end=end_offset,
        ))
        seg_id += 1

    for line in lines:
        line_start = cursor
        cursor += len(line) + 1  # +1 for the newline we split on
        header = _match_header(line)
        # Don't treat a "header-looking" line as a header if it's clearly a
        # Q&A section title rather than a speaker.
        if header and not re.search(r"|".join(_QA_MARKERS), line.lower()):
            flush(line_start)
            cur_speaker, cur_role = header
            cur_start = line_start
            buf = []
        else:
            buf.append(line)

    flush(cursor)
    return segments


def management_text(segments: list[Segment], section: str | None = None,
                    max_chars: int = 14000) -> str:
    """Concatenate management turns (optionally within one section)."""
    parts = []
    for s in segments:
        if s.role_type != "management":
            continue
        if section and s.section != section:
            continue
        parts.append(f"[{s.speaker} ({s.role})]\n{s.text}")
    out = "\n\n".join(parts)
    return out[:max_chars]


def analyst_questions(segments: list[Segment], max_chars: int = 6000) -> str:
    parts = [f"[{s.speaker}]\n{s.text}" for s in segments if s.role_type == "analyst"]
    return "\n\n".join(parts)[:max_chars]


def stats(segments: list[Segment]) -> dict:
    return {
        "segments": len(segments),
        "management_turns": sum(1 for s in segments if s.role_type == "management"),
        "analyst_turns": sum(1 for s in segments if s.role_type == "analyst"),
        "has_qa": any(s.section == "qa" for s in segments),
    }

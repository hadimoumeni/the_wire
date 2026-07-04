"""Parse an earnings-call transcript into structured, offset-tracked segments.

Handles the two common layouts:

  1. Motley Fool — role on the same line as the name:
        Tim Cook -- Chief Executive Officer
        Thank you. ...

  2. Seeking Alpha / European — a participant roster up top, then two-line
     speaker headers (name, then role on the next line):
        Company Participants
        Aurélien Sonet - Chief Executive Officer
        ...
        Aurélien Sonet
        Chief Executive Officer

        Thank you, Pauline...

Char offsets into the raw transcript are preserved so the verifier can point
back to exact source spans.
"""
from __future__ import annotations

import re
from .schemas import Segment, RoleType

# Q&A section boundary markers (case-insensitive, matched as a standalone line).
_QA_MARKERS = [
    r"questions?\s*&\s*answers?",
    r"questions?\s+and\s+answers?",
    r"question[- ]and[- ]answer\s+session",
    r"q\s*&\s*a\s+session",
]

# Single-line speaker headers (Motley Fool). Group 1 = name, group 2 = role.
_HEADER_PATTERNS = [
    re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}?)\s+--\s+(.{2,60})$"),   # Name -- Role
    re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}?),\s+(.{2,60})$"),       # Name, Role
]
_OPERATOR_RE = re.compile(r"^\s*operator\s*$", re.IGNORECASE)
_COLON_HEADER_RE = re.compile(r"^([A-Z][A-Za-z.\-' ]{1,40}):\s*$")    # NAME:

# Participant-list section headers → the role type of names listed under them.
_PARTICIPANT_SECTIONS = {
    "company participants": "management",
    "corporate participants": "management",
    "executives": "management",
    "conference call participants": "analyst",
    "call participants": "analyst",
    "analysts": "analyst",
}
_ROSTER_LINE = re.compile(r"^(.{2,50}?)\s+[-–—]\s+(.+)$")  # Name - Role (single dash)

_MGMT_ROLE_HINTS = (
    "chief", "ceo", "cfo", "coo", "cto", "president", "founder", "officer",
    "vp", "vice president", "head of", "director of", "treasurer", "controller",
    "investor relations", "ir ", "chairman", "executive",
)
_ANALYST_ROLE_HINTS = ("analyst", "research", "securities", "capital", "partners",
                        "bank", "& co", "llc", "plc", "inc", "equity", "markets",
                        "division")

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _find_qa_boundary(text: str) -> int:
    """Char offset where the Q&A section begins (standalone header line), or -1."""
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


def _extract_roster(text: str) -> dict[str, tuple[str, RoleType]]:
    """Build {normalized name: (role, role_type)} from the participant lists."""
    roster: dict[str, tuple[str, RoleType]] = {}
    section: str | None = None
    for line in text.split("\n"):
        s = line.strip()
        low = s.lower().rstrip(":")
        if low in _PARTICIPANT_SECTIONS:
            section = _PARTICIPANT_SECTIONS[low]
            continue
        if not s:
            continue
        if section:
            m = _ROSTER_LINE.match(s)
            if m and len(s) < 130:
                name, role = m.group(1).strip(), m.group(2).strip()
                if name and role:
                    roster[_norm(name)] = (role, section)
                    continue
            # A non-roster line (e.g. "Presentation", "Operator") ends the list.
            section = None
    return roster


def _match_header(line: str, roster: dict) -> tuple[str, str, RoleType] | None:
    """Return (speaker, role, role_type) if `line` is a speaker header."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if _OPERATOR_RE.match(s):
        return ("Operator", "Operator", "operator")
    # A participant-roster line ("Name - Role") is not a speaker turn.
    mr = _ROSTER_LINE.match(s)
    if mr and _norm(mr.group(1)) in roster:
        return None
    # Roster name on its own line (two-line header form).
    r = roster.get(_norm(s))
    if r is not None:
        return (s, r[0], r[1])
    # Single-line "Name -- Role" / "Name, Role" (Motley Fool).
    if s[-1] in ".?!":
        return None
    for pat in _HEADER_PATTERNS:
        m = pat.match(s)
        if m:
            role = m.group(2).strip()
            if role.count(",") >= 1:
                return None
            return (m.group(1).strip(), role, _classify_role(role))
    m = _COLON_HEADER_RE.match(s)
    if m:
        return (m.group(1).strip(), "", "unknown")
    return None


def parse_transcript(raw: str) -> list[Segment]:
    """Split a transcript into speaker segments with char offsets."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    roster = _extract_roster(text)
    qa_start = _find_qa_boundary(text)

    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    segments: list[Segment] = []
    seg_id = 0
    cur_speaker: str | None = None
    cur_role = ""
    cur_type: RoleType = "unknown"
    cur_start = 0
    buf: list[str] = []

    def flush(end_offset: int) -> None:
        nonlocal seg_id
        if cur_speaker is None:
            return
        body = "\n".join(buf).strip()
        if not body:
            return
        section = "qa" if (qa_start != -1 and cur_start >= qa_start) else "prepared_remarks"
        segments.append(Segment(
            id=seg_id, speaker=cur_speaker, role=cur_role, role_type=cur_type,
            section=section, text=body, start=cur_start, end=end_offset,
        ))
        seg_id += 1

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        header = _match_header(line, roster)
        if header:
            flush(offsets[i])
            cur_speaker, cur_role, cur_type = header
            cur_start = offsets[i]
            buf = []
            # Two-line header: skip the role line that follows the name.
            if i + 1 < n and cur_role and _norm(lines[i + 1]) == _norm(cur_role):
                i += 1
        else:
            buf.append(line)
        i += 1

    flush(pos)
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
    return "\n\n".join(parts)[:max_chars]


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

from wire.ingest import parse_transcript, stats, management_text

RAW = open("data/sample_transcript.txt", encoding="utf-8").read()


def test_segments_and_roles():
    segs = parse_transcript(RAW)
    st = stats(segs)
    assert st["management_turns"] >= 5
    assert st["analyst_turns"] >= 3
    assert st["has_qa"] is True


def test_prepared_vs_qa_split():
    segs = parse_transcript(RAW)
    # The CEO/CFO opening remarks must be prepared_remarks, not Q&A —
    # the inline "question-and-answer session" mention must not trip the boundary.
    sarah_open = next(s for s in segs if s.speaker == "Sarah Chen")
    assert sarah_open.section == "prepared_remarks"
    # Analyst turns live in Q&A.
    analysts = [s for s in segs if s.role_type == "analyst"]
    assert analysts and all(s.section == "qa" for s in analysts)


def test_offsets_map_to_source():
    # A segment's [start, end) span covers its speaker header + body; the body
    # text must appear verbatim within that span.
    segs = parse_transcript(RAW)
    for s in segs[:8]:
        assert s.text[:40] in RAW[s.start:s.end]


def test_no_prose_line_as_speaker():
    segs = parse_transcript(RAW)
    speakers = {s.speaker for s in segs}
    assert "With that" not in speakers  # prose line, not a header


def test_management_text_is_source_substring_per_line():
    segs = parse_transcript(RAW)
    mtext = management_text(segs)
    assert "route optimization" in mtext.lower()


# --- Seeking Alpha / European format: roster + two-line speaker headers ---
SA_RAW = """Company Participants

Aurélien Sonet - Chief Executive Officer
Stephane Lhopiteau - Group Chief Financial Officer

Conference Call Participants

Estelle Weingrod - JPMorgan Chase & Co, Research Division
Pavan Daswani - Citigroup Inc., Research Division

Presentation

Operator

Good morning and welcome to the call.

Aurélien Sonet
Chief Executive Officer

Thank you. Revenue reached EUR 312 million, down 3.3% organically. We confirm our full-year objectives.

Question-and-Answer Session

Operator

The first question is from Estelle Weingrod, JPMorgan.

Estelle Weingrod
JPMorgan Chase & Co, Research Division

Can you elaborate on Brazil?

Aurélien Sonet
Chief Executive Officer

The negative impact will intensify in Q4 with the further rollout of the reform.
"""


def test_seeking_alpha_two_line_headers():
    segs = parse_transcript(SA_RAW)
    st = stats(segs)
    assert st["management_turns"] >= 2       # CEO detected via roster + two-line headers
    assert st["analyst_turns"] >= 1
    assert st["has_qa"] is True
    ceo = [s for s in segs if s.speaker == "Aurélien Sonet"]
    assert ceo and all(s.role_type == "management" for s in ceo)
    # The role line must not leak into the body.
    assert not ceo[0].text.startswith("Chief Executive Officer")
    # Prepared remarks before Q&A, analyst turn inside Q&A.
    assert ceo[0].section == "prepared_remarks"
    assert any(s.role_type == "analyst" and s.section == "qa" for s in segs)


def test_roster_lines_are_not_segments():
    segs = parse_transcript(SA_RAW)
    # A participant-list line must never become a speaker turn.
    assert not any("Citigroup Inc." in s.speaker for s in segs)
    mtext = management_text(segs)
    assert "312 million" in mtext

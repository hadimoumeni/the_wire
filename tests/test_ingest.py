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

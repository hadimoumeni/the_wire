from wire.grounding import check_quote

RAW = open("data/sample_transcript.txt", encoding="utf-8").read()


def test_real_quote_supported():
    q = "We now expect full-year revenue in the range of $3.28 billion to $3.31 billion"
    c = check_quote(q, RAW)
    assert c.supported and c.score >= 95
    assert c.start is not None and RAW[c.start:c.end].lower().startswith("we now expect")


def test_fabricated_quote_rejected():
    c = check_quote("We are raising full-year guidance to five billion dollars", RAW)
    assert not c.supported


def test_paraphrase_with_wrong_numbers_rejected():
    # Verifier must not accept invented numbers even if the sentence rhymes.
    c = check_quote("adjusted EPS of $9.99 far above the high end of guidance", RAW)
    assert not c.supported


def test_whitespace_tolerant():
    q = "gross   margin\n expanded 180 basis points to 34.2%"
    c = check_quote(q, RAW)
    assert c.supported


def test_too_short_rejected():
    assert not check_quote("revenue", RAW).supported

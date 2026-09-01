from clauseforge.data.segment import segment_contract


def test_numbered_legal_sections_preserve_offsets() -> None:
    text = "Preamble\n1. TERM\nFirst clause.\n2. LAW\nSecond clause."

    segments = segment_contract(text)

    assert [segment.text for segment in segments] == [
        "Preamble",
        "1. TERM\nFirst clause.",
        "2. LAW\nSecond clause.",
    ]
    for segment in segments:
        assert text[segment.start_char : segment.end_char] == segment.text


def test_paragraph_fallback_is_conservative() -> None:
    text = "First paragraph.\n\nSecond paragraph."

    assert [item.text for item in segment_contract(text)] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_sentence_fallback_and_empty_text() -> None:
    assert [item.text for item in segment_contract("One sentence. Next sentence.")] == [
        "One sentence.",
        "Next sentence.",
    ]
    assert segment_contract("  \n") == []

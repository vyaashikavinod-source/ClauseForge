"""Rule-based legal-document candidate segmentation."""

from __future__ import annotations

import re
from itertools import pairwise

from clauseforge.data.models import TextSegment

_SECTION_START = re.compile(
    r"(?m)^(?=[ \t]*(?:(?:SECTION|ARTICLE)\s+[IVXLC\d]+\b|"
    r"\d+(?:\.\d+)*[.)]?\s+[A-Z]|[A-Z][A-Z &/\-]{3,}\s*$))"
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])(?:[ \t]+|\n+)(?=[A-Z0-9])")


def _trimmed_segment(
    text: str, start: int, end: int, strategy: str
) -> TextSegment | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    return TextSegment(text[start:end], start, end, strategy)


def _segments_from_boundaries(
    text: str, boundaries: list[int], strategy: str
) -> list[TextSegment]:
    points = sorted(set([0, *boundaries, len(text)]))
    segments: list[TextSegment] = []
    for start, end in pairwise(points):
        segment = _trimmed_segment(text, start, end, strategy)
        if segment is not None:
            segments.append(segment)
    return segments


def segment_contract(text: str) -> list[TextSegment]:
    """Prefer legal section markers, then paragraphs, then sentence boundaries."""
    if not text.strip():
        return []
    section_boundaries = [match.start() for match in _SECTION_START.finditer(text)]
    if len(section_boundaries) >= 2:
        return _segments_from_boundaries(text, section_boundaries, "legal_section")

    paragraph_boundaries = [match.end() for match in _PARAGRAPH_BREAK.finditer(text)]
    if paragraph_boundaries:
        return _segments_from_boundaries(text, paragraph_boundaries, "paragraph")

    sentence_boundaries = [match.end() for match in _SENTENCE_BREAK.finditer(text)]
    if sentence_boundaries:
        return _segments_from_boundaries(text, sentence_boundaries, "sentence")
    segment = _trimmed_segment(text, 0, len(text), "whole_document")
    return [segment] if segment is not None else []

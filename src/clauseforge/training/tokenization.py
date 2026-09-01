"""Tokenizer-aware encoding and explicit truncation measurement."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import ceil
from typing import Protocol

from clauseforge.training.templates import TrainingExample, render_example


class TokenizerProtocol(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TokenLengthReport:
    example_count: int
    minimum: int
    median: float
    p95: int
    maximum: int
    max_sequence_length: int
    truncated_count: int
    truncated_percentage: float
    truncation_by_category: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[max(0, ceil(len(values) * fraction) - 1)]


def analyze_token_lengths(
    examples: tuple[TrainingExample, ...],
    tokenizer: TokenizerProtocol,
    max_sequence_length: int,
) -> TokenLengthReport:
    if not examples:
        raise ValueError("token analysis requires at least one example")
    lengths = [len(tokenizer.encode(render_example(item))) for item in examples]
    ordered = sorted(lengths)
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    totals = Counter(item.target_label for item in examples)
    truncated = Counter(
        item.target_label
        for item, length in zip(examples, lengths, strict=True)
        if length > max_sequence_length
    )
    by_category = {
        label: {
            "examples": totals[label],
            "truncated": truncated[label],
            "percentage": round(100.0 * truncated[label] / totals[label], 6),
        }
        for label in sorted(totals)
    }
    count = sum(length > max_sequence_length for length in lengths)
    return TokenLengthReport(
        len(examples),
        min(lengths),
        median,
        _percentile(lengths, 0.95),
        max(lengths),
        max_sequence_length,
        count,
        round(100.0 * count / len(examples), 6),
        by_category,
    )

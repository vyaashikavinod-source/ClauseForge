"""Typed internal records for ClauseForge datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Trace a record back to its source and processing operation."""

    dataset: str
    dataset_version: str
    processing_version: str
    source_file: str
    source_document: str
    annotation_id: str | None = None
    original_start: int | None = None
    original_end: int | None = None
    original_text: str | None = None
    alignment: str | None = None

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContractRecord:
    """A normalized source contract."""

    contract_id: str
    source: str
    title: str
    raw_text: str
    metadata: JsonObject
    provenance: Provenance

    def to_dict(self) -> JsonObject:
        return {
            "contract_id": self.contract_id,
            "source": self.source,
            "title": self.title,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ClauseRecord:
    """An authoritative annotated clause span."""

    clause_id: str
    contract_id: str
    category: str
    text: str
    start_char: int
    end_char: int
    source: str
    provenance: Provenance

    def to_dict(self) -> JsonObject:
        return {
            "clause_id": self.clause_id,
            "contract_id": self.contract_id,
            "category": self.category,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source": self.source,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A deterministic, unlabeled candidate segment from contract text."""

    text: str
    start_char: int
    end_char: int
    strategy: str


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """An unlabeled, rule-derived candidate boundary within a contract."""

    segment_id: str
    contract_id: str
    text: str
    start_char: int
    end_char: int
    strategy: str

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Machine-readable description of one processed dataset build."""

    dataset_name: str
    source: str
    version: str
    processing_version: str
    created_at: str
    document_count: int
    clause_count: int
    category_count: int
    splits: dict[str, list[str]]
    checksums: dict[str, str]
    configuration: JsonObject = field(default_factory=dict)
    output_checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return asdict(self)

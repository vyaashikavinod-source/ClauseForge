"""Deterministic normalization of CUAD records into ClauseForge records."""

from __future__ import annotations

import hashlib

from clauseforge.data.cuad import CuadDataset
from clauseforge.data.models import ClauseRecord, ContractRecord, Provenance


class AnnotationAlignmentError(ValueError):
    """Raised when an authoritative source span cannot be aligned safely."""


def stable_id(kind: str, *parts: str) -> str:
    payload = "\x1f".join((kind, *parts)).encode()
    return f"{kind}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _anchored_whitespace_end(context: str, answer_text: str, start: int) -> int | None:
    """Match equal content with different whitespace runs at the supplied start."""
    source_index = start
    answer_index = 0
    while answer_index < len(answer_text):
        if source_index >= len(context):
            return None
        if answer_text[answer_index].isspace():
            if not context[source_index].isspace():
                return None
            while (
                answer_index < len(answer_text) and answer_text[answer_index].isspace()
            ):
                answer_index += 1
            while source_index < len(context) and context[source_index].isspace():
                source_index += 1
        else:
            if context[source_index] != answer_text[answer_index]:
                return None
            source_index += 1
            answer_index += 1
    return source_index


def align_annotation(
    context: str, answer_text: str, start: int
) -> tuple[str, int, str]:
    """Validate an offset without searching for or relocating the annotation."""
    if not answer_text:
        raise AnnotationAlignmentError("annotation text must not be empty")
    if start < 0 or start >= len(context):
        raise AnnotationAlignmentError(
            f"annotation start {start} is outside context length {len(context)}"
        )
    expected_end = start + len(answer_text)
    if expected_end <= len(context) and context[start:expected_end] == answer_text:
        return answer_text, expected_end, "exact"
    aligned_end = _anchored_whitespace_end(context, answer_text, start)
    if aligned_end is not None:
        return context[start:aligned_end], aligned_end, "whitespace_normalized"
    if expected_end > len(context):
        raise AnnotationAlignmentError(
            f"annotation end {expected_end} is outside context length {len(context)}"
        )
    raise AnnotationAlignmentError(
        f"annotation at [{start}:{expected_end}] does not match the source context"
    )


def normalize_cuad(
    dataset: CuadDataset, processing_version: str
) -> tuple[list[ContractRecord], list[ClauseRecord]]:
    """Normalize every CUAD answer span, failing on any unaligned annotation."""
    contracts: list[ContractRecord] = []
    clauses: list[ClauseRecord] = []
    for document in dataset.documents:
        separator = "\n\n"
        context_offsets: list[int] = []
        contexts: list[str] = []
        cursor = 0
        for paragraph in document.paragraphs:
            context_offsets.append(cursor)
            contexts.append(paragraph.context)
            cursor += len(paragraph.context) + len(separator)
        raw_text = separator.join(contexts)
        contract_id = stable_id("contract", "cuad", dataset.version, document.title)
        contract_provenance = Provenance(
            dataset="CUAD",
            dataset_version=dataset.version,
            processing_version=processing_version,
            source_file=str(dataset.source_path),
            source_document=document.title,
        )
        contracts.append(
            ContractRecord(
                contract_id=contract_id,
                source="CUAD",
                title=document.title,
                raw_text=raw_text,
                metadata={"paragraph_count": len(document.paragraphs)},
                provenance=contract_provenance,
            )
        )
        for paragraph_index, paragraph in enumerate(document.paragraphs):
            base_offset = context_offsets[paragraph_index]
            for question in paragraph.questions:
                for answer_index, answer in enumerate(question.answers):
                    try:
                        source_span, local_end, alignment = align_annotation(
                            paragraph.context, answer.text, answer.answer_start
                        )
                    except AnnotationAlignmentError as exc:
                        raise AnnotationAlignmentError(
                            f"document {document.title!r}, annotation "
                            f"{question.question_id!r}, answer {answer_index}: {exc}"
                        ) from exc
                    start = base_offset + answer.answer_start
                    end = base_offset + local_end
                    clause_id = stable_id(
                        "clause",
                        contract_id,
                        question.question_id,
                        str(answer_index),
                        str(start),
                        str(end),
                        answer.text,
                    )
                    clauses.append(
                        ClauseRecord(
                            clause_id=clause_id,
                            contract_id=contract_id,
                            category=question.category,
                            text=source_span,
                            start_char=start,
                            end_char=end,
                            source="CUAD",
                            provenance=Provenance(
                                dataset="CUAD",
                                dataset_version=dataset.version,
                                processing_version=processing_version,
                                source_file=str(dataset.source_path),
                                source_document=document.title,
                                annotation_id=question.question_id,
                                original_start=answer.answer_start,
                                original_end=answer.answer_start + len(answer.text),
                                original_text=answer.text,
                                alignment=alignment,
                            ),
                        )
                    )
    return contracts, clauses

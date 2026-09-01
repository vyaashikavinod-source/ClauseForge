"""Strict loader for the official CUAD v1 SQuAD 2.0-style JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CuadFormatError(ValueError):
    """Raised when a CUAD source file does not match the supported schema."""


@dataclass(frozen=True, slots=True)
class CuadAnswer:
    text: str
    answer_start: int


@dataclass(frozen=True, slots=True)
class CuadQuestion:
    question_id: str
    category: str
    answers: tuple[CuadAnswer, ...]
    is_impossible: bool


@dataclass(frozen=True, slots=True)
class CuadParagraph:
    context: str
    questions: tuple[CuadQuestion, ...]


@dataclass(frozen=True, slots=True)
class CuadDocument:
    title: str
    paragraphs: tuple[CuadParagraph, ...]


@dataclass(frozen=True, slots=True)
class CuadDataset:
    version: str
    documents: tuple[CuadDocument, ...]
    source_path: Path


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CuadFormatError(f"{location} must be a JSON object")
    return value


def _list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise CuadFormatError(f"{location} must be a JSON array")
    return value


def _string(value: object, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CuadFormatError(f"{location} must be a non-empty string")
    return value


def load_cuad(path: Path) -> CuadDataset:
    """Load a local `CUAD_v1.json`; this function never downloads data."""
    if not path.is_file():
        raise CuadFormatError(f"CUAD input file does not exist: {path}")
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CuadFormatError(f"Unable to read CUAD JSON at {path}: {exc}") from exc

    version = _string(root.get("version"), "root.version")
    raw_documents = _list(root.get("data"), "root.data")
    documents: list[CuadDocument] = []
    for document_index, raw_document in enumerate(raw_documents):
        doc_location = f"root.data[{document_index}]"
        document = _object(raw_document, doc_location)
        title = _string(document.get("title"), f"{doc_location}.title")
        raw_paragraphs = _list(document.get("paragraphs"), f"{doc_location}.paragraphs")
        if not raw_paragraphs:
            raise CuadFormatError(f"{doc_location}.paragraphs must not be empty")
        paragraphs: list[CuadParagraph] = []
        for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
            para_location = f"{doc_location}.paragraphs[{paragraph_index}]"
            paragraph = _object(raw_paragraph, para_location)
            context = _string(paragraph.get("context"), f"{para_location}.context")
            raw_questions = _list(paragraph.get("qas"), f"{para_location}.qas")
            questions: list[CuadQuestion] = []
            for question_index, raw_question in enumerate(raw_questions):
                question_location = f"{para_location}.qas[{question_index}]"
                question = _object(raw_question, question_location)
                question_id = _string(question.get("id"), f"{question_location}.id")
                category = _string(
                    question.get("question"), f"{question_location}.question"
                )
                is_impossible = question.get("is_impossible", False)
                if not isinstance(is_impossible, bool):
                    raise CuadFormatError(
                        f"{question_location}.is_impossible must be a boolean"
                    )
                raw_answers = _list(
                    question.get("answers", []), f"{question_location}.answers"
                )
                answers: list[CuadAnswer] = []
                for answer_index, raw_answer in enumerate(raw_answers):
                    answer_location = f"{question_location}.answers[{answer_index}]"
                    answer = _object(raw_answer, answer_location)
                    text = _string(answer.get("text"), f"{answer_location}.text")
                    answer_start = answer.get("answer_start")
                    if not isinstance(answer_start, int) or isinstance(
                        answer_start, bool
                    ):
                        raise CuadFormatError(
                            f"{answer_location}.answer_start must be an integer"
                        )
                    answers.append(CuadAnswer(text=text, answer_start=answer_start))
                if is_impossible and answers:
                    raise CuadFormatError(
                        f"{question_location} is impossible but contains answers"
                    )
                questions.append(
                    CuadQuestion(question_id, category, tuple(answers), is_impossible)
                )
            paragraphs.append(CuadParagraph(context, tuple(questions)))
        documents.append(CuadDocument(title, tuple(paragraphs)))

    if not documents:
        raise CuadFormatError("root.data must contain at least one document")
    return CuadDataset(version, tuple(documents), path.resolve())

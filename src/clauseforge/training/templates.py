"""One versioned source of truth for supervised prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_TEMPLATE_VERSION = "cuad-classification-v1"
SYSTEM_INSTRUCTION = (
    "You classify contract clauses using the authoritative ClauseForge/CUAD "
    "category taxonomy. Reply with exactly one category label."
)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    clause_id: str
    contract_id: str
    clause_text: str
    target_label: str
    split: str


def render_prompt(clause_text: str) -> str:
    return f"System:\n{SYSTEM_INSTRUCTION}\n\nUser:\n{clause_text}\n\nAssistant:\n"


def render_example(example: TrainingExample) -> str:
    return render_prompt(example.clause_text) + example.target_label

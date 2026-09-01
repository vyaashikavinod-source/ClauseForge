"""One versioned source of truth for supervised prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """A role-preserving message; clause content is always untrusted user data."""

    role: Literal["system", "user"]
    content: str


def render_messages(clause_text: str) -> tuple[PromptMessage, PromptMessage]:
    return (
        PromptMessage("system", SYSTEM_INSTRUCTION),
        PromptMessage("user", clause_text),
    )


def render_prompt(clause_text: str) -> str:
    system, user = render_messages(clause_text)
    return f"System:\n{system.content}\n\nUser:\n{user.content}\n\nAssistant:\n"


def render_example(example: TrainingExample) -> str:
    return render_prompt(example.clause_text) + example.target_label

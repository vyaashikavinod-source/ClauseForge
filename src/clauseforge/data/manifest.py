"""Checksums and dataset manifest construction."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from clauseforge.data.models import (
    ClauseRecord,
    ContractRecord,
    DatasetManifest,
    JsonObject,
)
from clauseforge.data.split import SplitMap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    source_path: Path,
    dataset_version: str,
    processing_version: str,
    contracts: list[ContractRecord],
    clauses: list[ClauseRecord],
    splits: SplitMap,
    configuration: JsonObject,
    output_checksums: dict[str, str],
    created_at: datetime | None = None,
) -> DatasetManifest:
    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
    return DatasetManifest(
        dataset_name="CUAD",
        source="The Atticus Project CUAD v1 SQuAD 2.0-style JSON",
        version=dataset_version,
        processing_version=processing_version,
        created_at=timestamp.isoformat().replace("+00:00", "Z"),
        document_count=len(contracts),
        clause_count=len(clauses),
        category_count=len({clause.category for clause in clauses}),
        splits=splits,
        checksums={str(source_path.resolve()): sha256_file(source_path)},
        configuration=configuration,
        output_checksums=output_checksums,
    )

from datetime import UTC, datetime
from pathlib import Path

from clauseforge.data.manifest import build_manifest, sha256_file
from tests.data.conftest import make_contract


def test_manifest_counts_configuration_and_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    contract = make_contract(1)

    manifest = build_manifest(
        source_path=source,
        dataset_version="1.0",
        processing_version="1.0.0",
        contracts=[contract],
        clauses=[],
        splits={"train": [contract.contract_id], "validation": [], "test": []},
        configuration={"seed": 42},
        output_checksums={},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert manifest.document_count == 1
    assert manifest.clause_count == 0
    assert manifest.checksums[str(source.resolve())] == sha256_file(source)
    assert manifest.created_at == "2026-01-01T00:00:00Z"

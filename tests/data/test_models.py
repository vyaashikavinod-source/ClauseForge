from clauseforge.data.models import ContractRecord, Provenance


def test_contract_serialization_is_explicit() -> None:
    record = ContractRecord(
        "contract-1",
        "CUAD",
        "example.txt",
        "Text",
        {"paragraph_count": 1},
        Provenance("CUAD", "1.0", "1.0.0", "source.json", "example.txt"),
    )

    serialized = record.to_dict()

    assert serialized["contract_id"] == "contract-1"
    assert serialized["provenance"] == {
        "dataset": "CUAD",
        "dataset_version": "1.0",
        "processing_version": "1.0.0",
        "source_file": "source.json",
        "source_document": "example.txt",
        "annotation_id": None,
        "original_start": None,
        "original_end": None,
        "original_text": None,
        "alignment": None,
    }

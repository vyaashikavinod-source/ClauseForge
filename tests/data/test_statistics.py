from clauseforge.data.statistics import build_statistics
from clauseforge.data.validate import ValidationReport
from tests.data.conftest import make_contract


def test_statistics_include_zero_annotation_contracts() -> None:
    contracts = [make_contract(1), make_contract(2)]
    splits = {
        "train": [contracts[0].contract_id],
        "validation": [contracts[1].contract_id],
        "test": [],
    }

    statistics = build_statistics(contracts, [], splits, ValidationReport((), 0))

    assert statistics["contract_count"] == 2
    assert statistics["clause_count"] == 0
    assert statistics["contracts_with_zero_annotations"] == [
        "contract-01",
        "contract-02",
    ]

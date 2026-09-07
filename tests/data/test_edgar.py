from __future__ import annotations

import asyncio
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, cast

import pytest
from scripts.run_edgar_ood import run as run_edgar_ood

from clauseforge.config import Settings
from clauseforge.data.edgar import (
    EdgarClient,
    RateLimiter,
    contract_filter,
    document_from_payload,
    extract_document_text,
    extract_submission_candidates,
    fetch_documents,
    prepare_documents,
    sample_review,
    stable_id,
    validate_ood,
)
from clauseforge.evaluation.ood import summarize_ood_predictions
from clauseforge.serving.providers.base import ProviderResult

FIXTURES = Path(__file__).parents[1] / "fixtures"


def metadata(
    name: str = "license.htm", accession: str = "0000000001-26-000001"
) -> dict[str, object]:
    return {
        "accession_number": accession,
        "cik": "1",
        "company_name": "Synthetic Example Corp",
        "filing_date": "2026-01-02",
        "form_type": "10-K",
        "exhibit_type": "EX-10.1",
        "document_name": name,
        "document_url": f"https://www.sec.gov/Archives/{name}",
        "metadata": {"fixture": True},
    }


def test_extracts_html_and_malformed_html() -> None:
    valid = extract_document_text(
        (FIXTURES / "edgar_contract.html").read_bytes(), "text/html"
    )
    malformed = extract_document_text(
        (FIXTURES / "edgar_malformed.html").read_bytes(), "text/html"
    )
    assert "LICENSE AGREEMENT" in valid
    assert "hidden" not in valid
    assert "EMPLOYMENT AGREEMENT" in malformed


def test_extracts_ex10_candidates_from_complete_submission() -> None:
    submission = b"""<DOCUMENT>
<TYPE>10-K
<FILENAME>report.htm
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-10.2
<FILENAME>license.htm
<DESCRIPTION>License Agreement
</DOCUMENT>"""
    candidates = extract_submission_candidates(
        submission,
        accession_number="0000000001-26-000001",
        cik="1",
        company_name="Synthetic Example Corp",
        filing_date="2026-01-02",
        form_type="10-K",
    )
    assert len(candidates) == 1
    assert candidates[0]["exhibit_type"] == "EX-10.2"
    assert str(candidates[0]["document_url"]).endswith("/license.htm")


def test_filter_is_conservative_and_explainable() -> None:
    assert contract_filter(
        exhibit_type="EX-10.1", document_name="agreement.htm", text="Agreement " * 80
    )[0]
    assert contract_filter(
        exhibit_type="EX-99", document_name="agreement.htm", text="Agreement " * 80
    ) == (False, "exhibit type is not EX-10")
    assert (
        contract_filter(exhibit_type="EX-10", document_name="notice.htm", text="short")[
            0
        ]
        is False
    )


def _write_raw(path: Path) -> None:
    payload = (FIXTURES / "edgar_contract.html").read_bytes()
    first = document_from_payload(
        metadata(), payload, "text/html", "2026-01-03T00:00:00Z"
    ).to_dict()
    duplicate = document_from_payload(
        metadata("copy.htm", "0000000001-26-000002"),
        payload,
        "text/html",
        "2026-01-03T00:00:00Z",
    ).to_dict()
    short = document_from_payload(
        metadata("short.htm", "0000000001-26-000003"),
        b"<p>Agreement.</p>",
        "text/html",
        "2026-01-03T00:00:00Z",
    ).to_dict()
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (first, duplicate, short)),
        encoding="utf-8",
    )


def test_prepare_deduplicates_segments_and_validates_offsets(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "processed"
    _write_raw(raw)
    stats = prepare_documents(
        raw, output, min_chars=300, created_at=datetime(2026, 1, 4, tzinfo=UTC)
    )
    assert stats["documents_downloaded"] == 3
    assert stats["documents_accepted"] == 1
    assert stats["documents_rejected"] == 2
    assert stats["duplicate_documents"] == 1
    assert cast(int, stats["segments_generated"]) >= 2
    documents = [
        json.loads(line)
        for line in (output / "documents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    segments = [
        json.loads(line)
        for line in (output / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_ood(documents, segments)
    for segment in segments:
        owner = documents[0]
        assert (
            owner["raw_text"][segment["start_char"] : segment["end_char"]]
            == segment["text"]
        )
        assert "category" not in segment and "split" not in segment
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["labels_assigned"] is False


def test_stable_ids_and_checksums_are_deterministic() -> None:
    payload = b"Agreement " * 100
    one = document_from_payload(
        metadata(), payload, "text/plain", "2026-01-01T00:00:00Z"
    )
    two = document_from_payload(
        metadata(), payload, "text/plain", "2026-01-02T00:00:00Z"
    )
    assert one.document_id == two.document_id
    assert one.checksum == two.checksum
    assert stable_id("x", "a", 1) == stable_id("x", "a", 1)


def test_missing_sec_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="required"):
        EdgarClient("", "")
    with pytest.raises(ValueError, match="valid"):
        EdgarClient("ClauseForge", "not-email")


def test_retry_logic_uses_cap() -> None:
    calls = 0
    sleeps: list[float] = []

    def failing(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("offline")

    client = EdgarClient(
        "ClauseForge research",
        "team@example.test",
        max_retries=2,
        opener=failing,
        sleeper=sleeps.append,
        requests_per_second=10,
    )
    with pytest.raises(RuntimeError, match="after retries"):
        client.get("https://www.sec.gov/example")
    assert calls == 3
    assert 1 in sleeps and 2 in sleeps


def test_rate_limiter_waits_for_remaining_interval() -> None:
    times = iter([0.0, 0.25, 1.0])
    sleeps: list[float] = []
    limiter = RateLimiter(1.0, clock=lambda: next(times), sleeper=sleeps.append)
    limiter.wait()
    limiter.wait()
    assert sleeps == [0.75]


def test_fetch_dry_run_and_resume(tmp_path: Path) -> None:
    payload = (FIXTURES / "edgar_contract.html").read_bytes()

    class Response:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/html"}

        def read(self) -> bytes:
            return payload

    client = EdgarClient(
        "ClauseForge",
        "team@example.test",
        opener=lambda *a, **k: Response(),
        sleeper=lambda _: None,
    )
    assert (
        fetch_documents(client, [metadata()], tmp_path, limit=1, dry_run=True)[
            "downloaded"
        ]
        == 0
    )
    assert fetch_documents(client, [metadata()], tmp_path, limit=1)["downloaded"] == 1
    result = fetch_documents(client, [metadata()], tmp_path, limit=1, resume=True)
    assert result["already_present"] == 1 and result["downloaded"] == 0


def test_review_sample_is_seeded(tmp_path: Path) -> None:
    source = tmp_path / "segments.jsonl"
    source.write_text(
        "".join(json.dumps({"segment_id": str(i)}) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    assert sample_review(source, first, count=4, seed=7) == 4
    sample_review(source, second, count=4, seed=7)
    assert first.read_bytes() == second.read_bytes()


def test_ood_metrics_never_claim_accuracy() -> None:
    result = summarize_ood_predictions(
        [
            {"prediction": "A", "confidence": 0.8},
            {"prediction": "invalid"},
            {"prediction": "A", "abstained": True, "error": "timeout"},
        ],
        frozenset({"A", "B"}),
    )
    assert result["taxonomy_valid_output_rate"] == pytest.approx(2 / 3)
    assert result["processing_failures"] == 1
    assert result["ground_truth_metrics_available"] is False
    assert "accuracy" not in result and "f1" not in result


class CategoryIdOodProvider:
    name = "category-id-fixture"
    model_id = "offline-fixture"
    provider_type = "transformer"
    is_mock = False

    async def classify(self, text: str) -> ProviderResult:
        category = "governing_law" if "law" in text else "invalid_id"
        return ProviderResult(category, category, None)

    async def close(self) -> None:
        return None


def test_edgar_summary_uses_category_id_output_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "segments.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"segment_id": str(index), "text": text})
            for index, text in enumerate(("governing law clause", "unknown clause"))
        ),
        encoding="utf-8",
    )
    settings = Settings(
        model_provider="transformer",
        model_artifact_manifest=tmp_path / "fixture-manifest.json",
        target_representation="category_id",
        target_representation_version="cuad-category-id-v1",
        prompt_template_version="cuad-classification-id-v2",
    )
    monkeypatch.setattr("scripts.run_edgar_ood.Settings.from_env", lambda: settings)
    monkeypatch.setattr(
        "scripts.run_edgar_ood.build_provider",
        lambda *args: CategoryIdOodProvider(),
    )
    result = asyncio.run(run_edgar_ood(source, tmp_path / "output"))
    assert result["prediction_distribution"] == {"governing_law": 1}
    assert result["taxonomy_valid_output_rate"] == 0.5
    assert result["invalid_output_rate"] == 0.5
    assert result["ground_truth_metrics_available"] is False
    assert result["target_representation"] == "category_id"
    assert "accuracy" not in result and "f1" not in result

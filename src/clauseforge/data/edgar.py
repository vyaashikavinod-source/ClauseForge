"""SEC EDGAR public-contract retrieval and unlabeled OOD preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean, median

from clauseforge.data.segment import segment_contract

PROCESSING_VERSION = "1.0.0"
SEC_SOURCE = "SEC EDGAR public filing exhibits"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SEC_COMPLETE_SUBMISSION = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession_dashed}.txt"
)
_CONTRACT_WORDS = (
    "agreement",
    "contract",
    "indenture",
    "lease",
    "license",
    "employment",
    "purchase",
    "merger",
    "amendment",
)
_SPACE = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n\s*\n(?:\s*\n)+")
_CHROME = re.compile(r"(?im)^\s*(?:document|type|sequence|filename|description):.*$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256_text(material)[:20]}"


@dataclass(frozen=True, slots=True)
class EdgarProvenance:
    """Public-source lineage for one EDGAR exhibit."""

    source: str
    accession_number: str
    document_url: str
    retrieved_at: str
    processing_version: str = PROCESSING_VERSION


@dataclass(frozen=True, slots=True)
class EdgarDocument:
    """Typed raw/extracted EDGAR exhibit record."""

    document_id: str
    accession_number: str
    cik: str
    company_name: str
    filing_date: str
    form_type: str
    exhibit_type: str
    document_name: str
    document_url: str
    raw_text: str
    source: str
    retrieved_at: str
    checksum: str
    normalized_checksum: str
    metadata: dict[str, object]
    provenance: EdgarProvenance
    selection_reason: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        return value


@dataclass(frozen=True, slots=True)
class OODClauseCandidate:
    """Unlabeled candidate span; this is not an authoritative clause label."""

    segment_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    source: str
    segmentation_strategy: str
    document_metadata: dict[str, object]
    provenance: dict[str, object]
    selection_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1
        elif not self.ignored and tag in {
            "p",
            "div",
            "br",
            "tr",
            "li",
            "h1",
            "h2",
            "h3",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.ignored:
            self.ignored -= 1
        elif not self.ignored and tag in {"p", "div", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def extract_document_text(payload: bytes, content_type: str = "") -> str:
    """Extract readable text from HTML or plain text without OCR."""
    decoded = payload.decode("utf-8", errors="replace").replace("\r\n", "\n")
    looks_html = "html" in content_type.lower() or bool(
        re.search(r"<\s*(?:html|body|div|p)\b", decoded, re.I)
    )
    if looks_html:
        parser = _TextExtractor()
        try:
            parser.feed(decoded)
            decoded = "".join(parser.parts)
        except Exception:
            decoded = re.sub(r"<[^>]*>", " ", decoded)
    decoded = _CHROME.sub("", decoded)
    lines = [_SPACE.sub(" ", line).strip() for line in decoded.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def contract_filter(
    *, exhibit_type: str, document_name: str, text: str, min_chars: int = 500
) -> tuple[bool, str]:
    """Conservatively retain likely contracts using explainable rules."""
    if len(text) < min_chars:
        return False, f"text shorter than {min_chars} characters"
    exhibit_match = exhibit_type.upper().startswith("EX-10")
    haystack = f"{document_name} {text[:5000]}".casefold()
    terms = sorted(term for term in _CONTRACT_WORDS if term in haystack)
    if not exhibit_match:
        return False, "exhibit type is not EX-10"
    if not terms:
        return False, "no contract-like title or heading terms"
    return True, f"EX-10 exhibit with contract-like terms: {', '.join(terms[:5])}"


class RateLimiter:
    """Monotonic minimum-interval limiter."""

    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < requests_per_second <= 10:
            raise ValueError(
                "requests_per_second must be greater than 0 and at most 10"
            )
        self.interval = 1.0 / requests_per_second
        self.clock = clock
        self.sleeper = sleeper
        self.last_request: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.last_request is not None:
            remaining = self.interval - (now - self.last_request)
            if remaining > 0:
                self.sleeper(remaining)
                now = self.clock()
        self.last_request = now


class EdgarClient:
    """Small synchronous SEC client with identification, throttling, and retry."""

    def __init__(
        self,
        user_agent: str,
        contact_email: str,
        *,
        requests_per_second: float = 2.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            not user_agent.strip()
            or not contact_email.strip()
            or "@" not in contact_email
        ):
            raise ValueError(
                "SEC_USER_AGENT and a valid SEC_CONTACT_EMAIL are required"
            )
        self.headers = {"User-Agent": f"{user_agent.strip()} {contact_email.strip()}"}
        self.timeout = timeout
        self.max_retries = max_retries
        self.opener = opener
        self.sleeper = sleeper
        self.limiter = RateLimiter(requests_per_second, sleeper=sleeper)

    def get(self, url: str) -> tuple[bytes, str]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            request = urllib.request.Request(url, headers=self.headers)
            try:
                response = self.opener(request, timeout=self.timeout)
                payload = response.read()  # type: ignore[attr-defined]
                content_type = response.headers.get("Content-Type", "")  # type: ignore[attr-defined]
                return bytes(payload), str(content_type)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if attempt == self.max_retries or not retryable:
                    break
                self.sleeper(min(2**attempt, 8))
        raise RuntimeError(f"SEC request failed after retries: {url}") from last_error


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}")
        yield value


def document_from_payload(
    metadata: Mapping[str, object], payload: bytes, content_type: str, retrieved_at: str
) -> EdgarDocument:
    text = extract_document_text(payload, content_type)
    accession = str(metadata["accession_number"])
    document_name = str(metadata["document_name"])
    document_url = str(metadata["document_url"])
    document_id = stable_id("edgar_doc", accession, document_name)
    checksum = hashlib.sha256(payload).hexdigest()
    raw_metadata = metadata.get("metadata")
    document_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    return EdgarDocument(
        document_id=document_id,
        accession_number=accession,
        cik=str(metadata["cik"]),
        company_name=str(metadata["company_name"]),
        filing_date=str(metadata["filing_date"]),
        form_type=str(metadata["form_type"]),
        exhibit_type=str(metadata["exhibit_type"]),
        document_name=document_name,
        document_url=document_url,
        raw_text=text,
        source=SEC_SOURCE,
        retrieved_at=retrieved_at,
        checksum=checksum,
        normalized_checksum=sha256_text(normalized_text(text)),
        metadata=document_metadata,
        provenance=EdgarProvenance(SEC_SOURCE, accession, document_url, retrieved_at),
    )


def fetch_documents(
    client: EdgarClient,
    candidates: Sequence[Mapping[str, object]],
    output: Path,
    *,
    limit: int,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Fetch explicit candidate exhibit URLs; never called at import time."""
    if limit < 1:
        raise ValueError("limit must be positive")
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "documents.jsonl"
    existing = (
        {str(row["document_id"]): row for row in _read_jsonl(records_path)}
        if resume
        else {}
    )
    failures: list[dict[str, object]] = []
    records = list(existing.values())
    selected = list(candidates[:limit])
    if dry_run:
        return {
            "planned": len(selected),
            "already_present": len(existing),
            "downloaded": 0,
            "failures": 0,
        }
    for item in selected:
        expected_id = stable_id(
            "edgar_doc", item["accession_number"], item["document_name"]
        )
        if expected_id in existing:
            continue
        try:
            payload, content_type = client.get(str(item["document_url"]))
            now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            records.append(
                document_from_payload(item, payload, content_type, now).to_dict()
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "document_url": str(item.get("document_url", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    _write_jsonl(records_path, records)
    _write_jsonl(output / "failures.jsonl", failures)
    return {
        "planned": len(selected),
        "already_present": len(existing),
        "downloaded": len(records) - len(existing),
        "failures": len(failures),
    }


def discover_exhibits(
    client: EdgarClient,
    ciks: Sequence[int],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    forms: Sequence[str] = ("10-K", "10-Q", "8-K"),
    exhibit_prefixes: Sequence[str] = ("EX-10",),
    limit: int = 50,
) -> list[dict[str, object]]:
    """Discover exhibit attachments through metadata and complete submissions."""
    lower = date.fromisoformat(start_date) if start_date else None
    upper = date.fromisoformat(end_date) if end_date else None
    results: list[dict[str, object]] = []
    for cik in ciks:
        payload, _ = client.get(SEC_SUBMISSIONS.format(cik=cik))
        root = json.loads(payload)
        recent = root.get("filings", {}).get("recent", {})
        keys = ("accessionNumber", "filingDate", "form")
        if not all(isinstance(recent.get(key), list) for key in keys):
            raise ValueError(f"malformed SEC submissions response for CIK {cik}")
        for accession, filing_date, form in zip(
            *(recent[key] for key in keys), strict=True
        ):
            filed = date.fromisoformat(str(filing_date))
            if (lower and filed < lower) or (upper and filed > upper):
                continue
            form_text = str(form).upper()
            if form_text not in {value.upper() for value in forms}:
                continue
            accession_compact = str(accession).replace("-", "")
            submission_url = SEC_COMPLETE_SUBMISSION.format(
                cik=cik,
                accession=accession_compact,
                accession_dashed=accession,
            )
            submission, _ = client.get(submission_url)
            results.extend(
                extract_submission_candidates(
                    submission,
                    accession_number=str(accession),
                    cik=str(cik),
                    company_name=str(root.get("name", "")),
                    filing_date=filed.isoformat(),
                    form_type=form_text,
                    exhibit_prefixes=exhibit_prefixes,
                )
            )
            if len(results) >= limit:
                return sorted(
                    results[:limit],
                    key=lambda item: (
                        str(item["filing_date"]),
                        str(item["accession_number"]),
                        str(item["document_name"]),
                    ),
                )
    return sorted(
        results,
        key=lambda item: (
            str(item["filing_date"]),
            str(item["accession_number"]),
            str(item["document_name"]),
        ),
    )


def extract_submission_candidates(
    payload: bytes,
    *,
    accession_number: str,
    cik: str,
    company_name: str,
    filing_date: str,
    form_type: str,
    exhibit_prefixes: Sequence[str] = ("EX-10",),
) -> list[dict[str, object]]:
    """Parse SEC complete-submission headers without extracting exhibit bodies."""
    text = payload.decode("utf-8", errors="replace")
    blocks = re.findall(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", text)
    accession_compact = accession_number.replace("-", "")
    candidates: list[dict[str, object]] = []
    for block in blocks:
        type_match = re.search(r"(?im)^<TYPE>\s*([^\r\n<]+)", block)
        name_match = re.search(r"(?im)^<FILENAME>\s*([^\r\n<]+)", block)
        description_match = re.search(r"(?im)^<DESCRIPTION>\s*([^\r\n<]+)", block)
        if type_match is None or name_match is None:
            continue
        exhibit_type = type_match.group(1).strip().upper()
        if not any(
            exhibit_type.startswith(prefix.upper()) for prefix in exhibit_prefixes
        ):
            continue
        document_name = name_match.group(1).strip()
        candidates.append(
            {
                "accession_number": accession_number,
                "cik": cik,
                "company_name": company_name,
                "filing_date": filing_date,
                "form_type": form_type,
                "exhibit_type": exhibit_type,
                "document_name": document_name,
                "document_url": SEC_ARCHIVES.format(
                    cik=cik,
                    accession=accession_compact,
                    document=document_name,
                ),
                "metadata": {
                    "description": description_match.group(1).strip()
                    if description_match
                    else ""
                },
            }
        )
    return candidates


def _distribution(values: list[int]) -> dict[str, object]:
    return (
        {
            "min": min(values),
            "max": max(values),
            "mean": round(mean(values), 3),
            "median": round(median(values), 3),
        }
        if values
        else {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    )


def prepare_documents(
    raw_path: Path,
    output: Path,
    *,
    min_chars: int = 500,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Filter, deduplicate, segment, validate, and describe raw EDGAR records."""
    rows = list(_read_jsonl(raw_path))
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    duplicate_count = 0
    seen_ids: set[str] = set()
    seen_checksums: set[str] = set()
    for row in rows:
        identifier = str(row["document_id"])
        checksums = {str(row["checksum"]), str(row["normalized_checksum"])}
        if identifier in seen_ids or seen_checksums.intersection(checksums):
            duplicate_count += 1
            row["rejection_reason"] = "duplicate identifier or content checksum"
            rejected.append(row)
            continue
        seen_ids.add(identifier)
        seen_checksums.update(checksums)
        keep, reason = contract_filter(
            exhibit_type=str(row["exhibit_type"]),
            document_name=str(row["document_name"]),
            text=str(row["raw_text"]),
            min_chars=min_chars,
        )
        row["selection_reason" if keep else "rejection_reason"] = reason
        (accepted if keep else rejected).append(row)
    segments: list[dict[str, object]] = []
    for document in accepted:
        text = str(document["raw_text"])
        for segment in segment_contract(text):
            raw_provenance = document["provenance"]
            if not isinstance(raw_provenance, dict):
                raise ValueError("malformed EDGAR provenance")
            candidate = OODClauseCandidate(
                segment_id=stable_id(
                    "edgar_seg",
                    document["document_id"],
                    segment.start_char,
                    segment.end_char,
                    segment.text,
                ),
                document_id=str(document["document_id"]),
                text=segment.text,
                start_char=segment.start_char,
                end_char=segment.end_char,
                source=SEC_SOURCE,
                segmentation_strategy=segment.strategy,
                document_metadata={
                    key: document[key]
                    for key in (
                        "accession_number",
                        "cik",
                        "company_name",
                        "filing_date",
                        "form_type",
                        "exhibit_type",
                        "document_name",
                        "document_url",
                    )
                },
                provenance=dict(raw_provenance),
                selection_metadata={"selection_reason": document["selection_reason"]},
            )
            segments.append(candidate.to_dict())
    validate_ood(accepted, segments)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "documents.jsonl", accepted)
    _write_jsonl(output / "segments.jsonl", segments)
    _write_jsonl(output / "rejected.jsonl", rejected)
    _write_jsonl(output / "failures.jsonl", [])
    exhibit_counts = Counter(str(row["exhibit_type"]) for row in accepted)
    years = Counter(str(row["filing_date"])[:4] for row in accepted)
    companies = Counter(str(row["company_name"]) for row in accepted)
    strategies = Counter(str(row["segmentation_strategy"]) for row in segments)
    statistics: dict[str, object] = {
        "documents_downloaded": len(rows),
        "documents_accepted": len(accepted),
        "documents_rejected": len(rejected),
        "duplicate_documents": duplicate_count,
        "segments_generated": len(segments),
        "segments_per_document": _distribution(
            [
                sum(
                    segment["document_id"] == row["document_id"] for segment in segments
                )
                for row in accepted
            ]
        ),
        "document_length_distribution": _distribution(
            [len(str(row["raw_text"])) for row in accepted]
        ),
        "segment_length_distribution": _distribution(
            [len(str(row["text"])) for row in segments]
        ),
        "exhibit_type_counts": dict(sorted(exhibit_counts.items())),
        "filing_year_counts": dict(sorted(years.items())),
        "company_counts": dict(sorted(companies.items())),
        "segmentation_strategy_counts": dict(sorted(strategies.items())),
        "fetch_failures": 0,
        "parsing_failures": 0,
        "filter_statistics": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "duplicates": duplicate_count,
        },
    }
    _write_json(output / "statistics.json", statistics)
    now = (
        (created_at or datetime.now(UTC))
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    dates = sorted(str(row["filing_date"]) for row in accepted)
    manifest: dict[str, object] = {
        "dataset_name": "SEC EDGAR contract exhibits OOD",
        "source": SEC_SOURCE,
        "processing_version": PROCESSING_VERSION,
        "retrieval_window": {
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
        },
        "document_count": len(accepted),
        "segment_count": len(segments),
        "duplicate_count": duplicate_count,
        "filter_statistics": statistics["filter_statistics"],
        "exhibit_type_distribution": statistics["exhibit_type_counts"],
        "company_count": len(companies),
        "date_range": {
            "min": dates[0] if dates else None,
            "max": dates[-1] if dates else None,
        },
        "checksums": {
            "raw_input": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "documents": hashlib.sha256(
                (output / "documents.jsonl").read_bytes()
            ).hexdigest(),
            "segments": hashlib.sha256(
                (output / "segments.jsonl").read_bytes()
            ).hexdigest(),
        },
        "configuration": {
            "minimum_text_characters": min_chars,
            "labels_assigned": False,
            "purpose": "out-of-distribution evaluation",
        },
        "creation_timestamp": now,
    }
    _write_json(output / "manifest.json", manifest)
    return statistics


def validate_ood(
    documents: Sequence[Mapping[str, object]], segments: Sequence[Mapping[str, object]]
) -> None:
    """Enforce OOD integrity without CUAD labels or split membership."""
    by_id: dict[str, Mapping[str, object]] = {}
    checksums: set[str] = set()
    for document in documents:
        document_id = str(document["document_id"])
        text = str(document["raw_text"])
        if not text or document_id in by_id:
            raise ValueError("empty or duplicate EDGAR document")
        if sha256_text(normalized_text(text)) != document["normalized_checksum"]:
            raise ValueError(f"normalized checksum mismatch: {document_id}")
        if str(document["normalized_checksum"]) in checksums:
            raise ValueError("duplicate EDGAR document content")
        checksums.add(str(document["normalized_checksum"]))
        by_id[document_id] = document
    segment_ids: set[str] = set()
    for segment in segments:
        if "category" in segment or "split" in segment:
            raise ValueError(
                "EDGAR OOD segments must not contain CUAD labels or splits"
            )
        owner = by_id.get(str(segment["document_id"]))
        if owner is None:
            raise ValueError("segment owner is missing")
        raw_start, raw_end = segment["start_char"], segment["end_char"]
        if not isinstance(raw_start, int) or not isinstance(raw_end, int):
            raise ValueError("EDGAR segment offsets must be integers")
        start, end = raw_start, raw_end
        text = str(segment["text"])
        if (
            not 0 <= start < end <= len(str(owner["raw_text"]))
            or str(owner["raw_text"])[start:end] != text
        ):
            raise ValueError("invalid EDGAR segment span")
        expected = stable_id("edgar_seg", segment["document_id"], start, end, text)
        if segment["segment_id"] != expected or expected in segment_ids:
            raise ValueError("unstable or duplicate EDGAR segment ID")
        segment_ids.add(expected)


def sample_review(input_path: Path, output: Path, *, count: int, seed: int) -> int:
    import random

    if count < 1:
        raise ValueError("count must be positive")
    rows = list(_read_jsonl(input_path))
    chosen = random.Random(seed).sample(rows, min(count, len(rows)))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, chosen)
    return len(chosen)


def _iso_date(value: str) -> str:
    return date.fromisoformat(value).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or validate SEC EDGAR OOD records"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--min-chars", type=int, default=500)
    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True, type=Path)
    sample = sub.add_parser("sample")
    sample.add_argument("--input", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--count", type=int, default=50)
    sample.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_documents(args.input, args.output, min_chars=args.min_chars)
        elif args.command == "validate":
            root = args.input
            validate_ood(
                list(_read_jsonl(root / "documents.jsonl")),
                list(_read_jsonl(root / "segments.jsonl")),
            )
        else:
            sample_review(args.input, args.output, count=args.count, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"EDGAR command failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SEC EDGAR out-of-distribution data

ClauseForge uses public SEC EDGAR contract exhibits only as an **unlabeled
out-of-distribution (OOD) evaluation source**. EDGAR records never enter CUAD
train, validation, or test splits, and the pipeline never assigns CUAD labels.
Retained segments are candidates for later prediction and human review, not
ground-truth clauses.

## Retrieval and SEC etiquette

Downloads are explicit. Nothing accesses the network during import, tests, or
installation. Discovery uses
`https://data.sec.gov/submissions/CIK##########.json`; exhibit bodies use
`https://www.sec.gov/Archives/edgar/data/...`.

```bash
export SEC_USER_AGENT="ClauseForge research"
export SEC_CONTACT_EMAIL="team@example.org"
python scripts/fetch_edgar_contracts.py --output data/raw/edgar --cik 320193 --limit 25 --resume
```

Use a real application name and monitored address. Values come only from the
environment. The default is two requests per second, with an enforced ceiling
of ten, a 30-second timeout, three capped retries, exponential backoff, and
deterministic resume keys. Date, form, exhibit, dry-run, and resume options make
retrieval bounded. Review current SEC guidance before a live run.

Submissions metadata identifies recent filings; discovery then reads the SEC
complete-submission text and selects explicit EX-10 attachment headers. A
reviewed JSON candidate array can instead be supplied via `--candidates`.

## Preparation and validation

The tolerant HTML/plain-text extractor removes script/style content, obvious
filing chrome, and redundant whitespace. OCR is absent. Selection requires an
EX-10 exhibit, minimum length, and contract-like title/heading terms. Decisions
store explainable reasons and remain heuristic. Exact-byte and normalized-text
SHA-256 checksums detect duplicates. The existing legal segmenter produces
stable unlabeled candidates with exact offsets and provenance.

```bash
python scripts/prepare_edgar.py --input data/raw/edgar/documents.jsonl --output data/processed/edgar/1.0.0
python scripts/validate_edgar.py --input data/processed/edgar/1.0.0
python scripts/sample_edgar_review.py --input data/processed/edgar/1.0.0/segments.jsonl --count 50 --seed 42 --output eval/review/edgar_sample.jsonl
```

## Metrics and limitations

Valid future metrics are taxonomy-valid and invalid-output rates, prediction
distribution, genuine-score confidence distribution, abstention rate, and
processing failures. Accuracy, precision, recall, and F1 are invalid without
authoritative labels.

Filtering cannot prove every retained exhibit is a contract. Complex table
layout may be lost, scans are unsupported, and metadata discovery does not
expose all attachments. Public filings may contain personal information and
still require careful handling. Human review and real-model evaluation remain
future work; this phase makes no performance claim.

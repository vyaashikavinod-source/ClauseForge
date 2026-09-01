# Data directories

No datasets are committed. Phase 1 supports a locally supplied public CUAD v1
SQuAD 2.0-style JSON file (`CUAD_v1.json`). Downloads are intentionally manual.

- `raw/` is reserved for immutable source data.
- `interim/` is reserved for intermediate transformations.
- `processed/` is reserved for validated, model-ready artifacts.

The contents of these directories are ignored by Git. Only `.gitkeep` markers
are tracked. Use public or explicitly approved synthetic data, record its
provenance in a future manifest, and never place client or proprietary data in
the repository.

## CUAD layout

A recommended local layout is:

```text
data/raw/cuad/CUAD_v1.json
data/processed/cuad/1.0.0/
├── contracts.jsonl
├── clauses.jsonl
├── segments.jsonl
├── splits.json
├── manifest.json
└── statistics.json
```

Run preparation with:

```bash
python -m clauseforge.data.prepare \
  --input data/raw/cuad/CUAD_v1.json \
  --output data/processed/cuad/1.0.0
```

The loader supports the official SQuAD 2.0-style JSON because it carries the
contract context, CUAD category question, annotation identifier, answer text,
and answer start offset in one auditable format. Master CSV/XLSX, PDF, and TXT
distributions are not parsed in Phase 1.

The generated manifest records the source SHA-256 checksum, CUAD and processing
versions, split configuration, record counts, and generated-file checksums.

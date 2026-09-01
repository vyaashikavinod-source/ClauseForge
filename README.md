# ClauseForge

**Status: In Development**

ClauseForge is a planned contract-clause intelligence system. The project is
intended to support clause classification, grounded risk analysis, model
evaluation, quantized deployment, and production-style inference. These
capabilities are not implemented yet.

## Current status

The repository contains the Phase 0 engineering foundation and a Phase 1 local
data pipeline for the public CUAD v1 dataset. The pipeline parses CUAD's SQuAD
2.0-style JSON, preserves authoritative annotation spans and provenance,
validates normalized records, creates contract-level splits, and writes
manifests and statistics.

ClauseForge does not currently fine-tune models, analyze contracts, run model
benchmarks, quantize models, or provide a deployed inference service.

## Architecture direction

The target system will separate data preparation, training, evaluation,
quantization, and serving concerns. Shared configuration and observability will
live in the core Python package. See [docs/architecture.md](docs/architecture.md)
for the planned component boundaries.

## Development roadmap

- [x] Phase 0: Repository foundation and development tooling
- [x] Phase 1: Public dataset ingestion and validation
- [ ] Phase 2: Clause classification baseline
- [ ] Phase 3: Grounded risk analysis
- [ ] Phase 4: Evaluation and frontier-model comparison
- [ ] Phase 5: Quantization and inference serving
- [ ] Phase 6: Regression evaluation and production hardening

Only one phase is implemented and validated at a time. Later phases remain
planning targets, not claims of working functionality.

## Local development

Python 3.11 is the supported development and CI version.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make check
```

Configuration is read from environment variables. Copy `.env.example` only as
a reference; the application does not automatically load `.env` files.

## Preparing CUAD

Download CUAD v1 yourself from the public [Atticus Project CUAD
repository](https://github.com/TheAtticusProject/cuad). Dataset downloads are
never performed implicitly and must remain under an ignored data directory.

```bash
python -m clauseforge.data.prepare \
  --input data/raw/cuad/CUAD_v1.json \
  --output data/processed/cuad/1.0.0
```

Add `--dry-run` to validate and summarize the input without writing files. The
committed `scripts/prepare_cuad.py` wrapper provides the same command.

Outputs are deterministic JSON/JSONL where practical: `contracts.jsonl`,
`clauses.jsonl`, unlabeled `segments.jsonl`, `splits.json`, `manifest.json`, and
`statistics.json`. Contract, clause, and segment IDs are content-derived. The
manifest creation timestamp is run metadata; checksums cover the source and
deterministic generated files.

Splitting occurs at the contract level with a deterministic seed and default
80/10/10 ratios. This prevents annotations from one contract appearing in more
than one split. CUAD annotation text and offsets remain ground truth; the
pipeline does not invent, rewrite, or relocate labels.

## Data and security

Use only public or explicitly approved synthetic data. Raw, interim, and
processed data artifacts are ignored by Git except for directory markers and
documentation. Never commit credentials, client information, proprietary data,
model weights, or generated evaluation outputs.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

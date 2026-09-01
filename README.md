# ClauseForge

**Status: In Development**

ClauseForge is a planned contract-clause intelligence system. The project is
intended to support clause classification, grounded risk analysis, model
evaluation, quantized deployment, and production-style inference. These
capabilities are not implemented yet.

## Current status

The repository currently contains only the Phase 0 engineering foundation:
package scaffolding, environment-based configuration, structured logging,
quality tooling, tests, and continuous integration.

ClauseForge does not currently fine-tune models, analyze contracts, run model
benchmarks, quantize models, or provide a deployed inference service.

## Architecture direction

The target system will separate data preparation, training, evaluation,
quantization, and serving concerns. Shared configuration and observability will
live in the core Python package. See [docs/architecture.md](docs/architecture.md)
for the planned component boundaries.

## Development roadmap

- [x] Phase 0: Repository foundation and development tooling
- [ ] Phase 1: Public dataset ingestion and validation
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

## Data and security

Use only public or explicitly approved synthetic data. Raw, interim, and
processed data artifacts are ignored by Git except for directory markers and
documentation. Never commit credentials, client information, proprietary data,
model weights, or generated evaluation outputs.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

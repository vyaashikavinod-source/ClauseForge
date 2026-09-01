"""Document the supported evaluation entry point for transformer adapters."""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "Load a trained TransformerClassifier and pass it to "
        "clauseforge.evaluation.runner.evaluate_classifier. Phase 3A creates "
        "only a smoke adapter, not a benchmark candidate."
    )


if __name__ == "__main__":
    raise SystemExit(main())

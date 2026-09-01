"""Measure real training/validation lengths with a provisioned tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from clauseforge.training.checkpoints import write_json
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.tokenization import analyze_token_lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        local_files_only=True,
    )
    dataset = build_training_dataset(args.data)
    train = analyze_token_lengths(
        dataset.train, tokenizer, args.max_sequence_length
    ).to_dict()
    validation = analyze_token_lengths(
        dataset.validation, tokenizer, args.max_sequence_length
    ).to_dict()
    result = {
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "max_sequence_length": args.max_sequence_length,
        "selection_splits": ["train", "validation"],
        "test_used": False,
        "train": train,
        "validation": validation,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

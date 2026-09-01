"""Benchmark a configured ClauseForge endpoint with bounded async load."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from clauseforge.serving.benchmark import load_test, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--provider", default="unknown")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    observations = asyncio.run(
        load_test(
            args.base_url, args.requests, args.concurrency, args.timeout, args.warmup
        )
    )
    result = summarize(
        observations,
        time.monotonic() - started,
        provider=args.provider,
        model=args.model,
        backend=args.backend,
        is_mock=args.development,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

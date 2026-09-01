"""Explicit, identified, rate-limited SEC EDGAR exhibit downloader."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from clauseforge.data.edgar import EdgarClient, discover_exhibits, fetch_documents


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Fetch a small public SEC EDGAR contract-exhibit set"
    )
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--limit", type=int, default=50)
    value.add_argument("--cik", action="append", type=int, default=[])
    value.add_argument(
        "--candidates", type=Path, help="Auditable JSON array of exhibit metadata/URLs"
    )
    value.add_argument("--start-date")
    value.add_argument("--end-date")
    value.add_argument("--form", action="append", default=[])
    value.add_argument("--exhibit-prefix", action="append", default=[])
    value.add_argument("--requests-per-second", type=float, default=2.0)
    value.add_argument("--timeout", type=float, default=30.0)
    value.add_argument("--max-retries", type=int, default=3)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--resume", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        client = EdgarClient(
            os.environ.get("SEC_USER_AGENT", ""),
            os.environ.get("SEC_CONTACT_EMAIL", ""),
            requests_per_second=args.requests_per_second,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        if args.candidates:
            candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
            if not isinstance(candidates, list):
                raise ValueError("--candidates must contain a JSON array")
        elif args.cik:
            candidates = discover_exhibits(
                client,
                args.cik,
                start_date=args.start_date,
                end_date=args.end_date,
                forms=args.form or ("10-K", "10-Q", "8-K"),
                exhibit_prefixes=args.exhibit_prefix or ("EX-10",),
                limit=args.limit,
            )
        else:
            raise ValueError("provide --candidates or at least one --cik")
        print(
            json.dumps(
                fetch_documents(
                    client,
                    candidates,
                    args.output,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    resume=args.resume,
                ),
                sort_keys=True,
            )
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"EDGAR fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

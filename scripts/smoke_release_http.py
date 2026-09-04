"""Synthetic HTTP smoke checks for an isolated real release container."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def run(url: str, artifact_id: str, step: int) -> dict[str, object]:
    def request(
        path: str, payload: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(
            url.rstrip("/") + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": str(uuid.uuid4()),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    for path in ("/health", "/", "/docs", "/metrics"):
        if request(path)[0] != 200:
            raise ValueError(f"smoke endpoint failed: {path}")
    status, _, raw = request("/ready")
    ready = json.loads(raw)
    if (
        status != 200
        or ready.get("artifact_id") != artifact_id
        or ready.get("checkpoint_step") != step
    ):
        raise ValueError("wrong or unavailable release artifact")
    cached = False
    for index in range(2):
        status, headers, raw = request(
            "/v1/classify",
            {"text": "This agreement is governed by the laws of Delaware."},
        )
        result = json.loads(raw)
        if (
            status != 200
            or result.get("processing", {}).get("is_mock") is not False
            or result.get("provider") != "active-trained-candidate"
        ):
            raise ValueError("real classification smoke failed")
        lowered = {key.lower(): value for key, value in headers.items()}
        if (
            not lowered.get("x-request-id")
            or lowered.get("x-content-type-options") != "nosniff"
        ):
            raise ValueError("request IDs/security middleware missing")
        if index == 1:
            cached = result["processing"]["cache_hit"] is True
    if not cached:
        raise ValueError("exact cache smoke failed")
    limited = any(
        request("/v1/classify", {"text": "Synthetic smoke clause for rate limiting."})[
            0
        ]
        == 429
        for _ in range(10)
    )
    if not limited:
        raise ValueError("isolated smoke rate limit (8/minute) not enforced")
    return {
        "label": "REAL CONTAINER SYNTHETIC SMOKE — NOT MODEL PERFORMANCE",
        "artifact_id": artifact_id,
        "checkpoint_step": step,
        "passed": True,
        "cache_verified": cached,
        "rate_limit_verified": limited,
        "test_evaluated": False,
        "logs_verified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("smoke evidence already exists")
    report = run(args.url, args.artifact_id, args.checkpoint_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

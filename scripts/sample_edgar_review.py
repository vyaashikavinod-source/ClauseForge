"""Deterministically sample unlabeled EDGAR candidates for human review."""

from clauseforge.data.edgar import main

if __name__ == "__main__":
    raise SystemExit(main(["sample", *__import__("sys").argv[1:]]))

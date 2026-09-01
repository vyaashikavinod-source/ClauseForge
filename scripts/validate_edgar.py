"""Explicit SEC EDGAR OOD validation entry point."""

from clauseforge.data.edgar import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *__import__("sys").argv[1:]]))

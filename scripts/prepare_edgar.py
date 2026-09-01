"""Explicit SEC EDGAR OOD preparation entry point."""

from clauseforge.data.edgar import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *__import__("sys").argv[1:]]))

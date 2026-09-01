.PHONY: install lint format-check typecheck test check

install:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

typecheck:
	python -m mypy

test:
	python -m pytest

check: lint format-check typecheck test

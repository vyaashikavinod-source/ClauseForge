.PHONY: install lint format-check typecheck test check serve docker-build docker-smoke readiness

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

serve:
	uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t clauseforge:local .

docker-smoke: docker-build
	docker run --rm -p 8000:8000 clauseforge:local

readiness:
	python scripts/check_release_readiness.py --json

.PHONY: lint fmt test build up down logs clean run

lint:
	ruff check .

fmt:
	ruff format .

test:
	pytest tests/ -v

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

run:
	python -m src.cli

run-http:
	python -m src.cli --http

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

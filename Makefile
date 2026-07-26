# ══════════════════════════════════════════════════════════════════════════════
# Memex — Development Makefile
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: help setup dev test lint fmt clean build up down logs ps e2e

# ── Default ──────────────────────────────────────────────────────────────────
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "\033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ── Project setup ────────────────────────────────────────────────────────────
setup: ## Bootstrap everything (Docker + models + deps)
	./setup.sh

dev: ## Install dev dependencies
	uv sync --extra dev --extra test

# ── Testing ──────────────────────────────────────────────────────────────────
test: ## Run unit tests
	uv run pytest tests/unit/ -q

test-all: ## Run all tests (unit + integration)
	uv run pytest tests/ -q

e2e: ## Run end-to-end test
	uv run python scripts/test_e2e.py

lint: ## Lint code
	uv run ruff check .

fmt: ## Auto-format + fix lint
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run mypy
	uv run mypy .

# ── Docker ───────────────────────────────────────────────────────────────────
up: ## Start backend services
	docker compose up -d

down: ## Stop backend services
	docker compose down

build: ## Rebuild custom images
	docker compose build

ps: ## Show service status
	docker compose ps

logs: ## Tail all logs
	docker compose logs -f

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov/ dist/ build/ *.egg-info

docker-clean: ## Stop containers and remove volumes (DESTRUCTIVE)
	docker compose down -v

docker-prune: ## Remove unused Docker data (images, volumes, networks)
	docker system prune -f

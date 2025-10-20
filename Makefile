# =======================================
# KitchenWatch Development Makefile
# =======================================

PYTHON_VERSION := 3.12
SRC_DIR=src/kitchenwatch
TEST_DIR=tests
VENV := .venv-local
UV_RUN := $(VENV)/bin/python -m uv run --active

# Default target
.DEFAULT_GOAL := help

# --- Setup ---------------------------------------------------------

## Create or update the local virtual environment
venv:
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install "uv>=0.9.5"
	$(VENV)/bin/python -m uv sync --group dev --active
	$(VENV)/bin/python -m uv run --active pre-commit install


## Reinstall all dev dependencies cleanly
reinstall:
	rm -rf $(VENV)
	make venv


## Print env info
env-info:
	@echo "=== Environment Info ==="
	@echo "User: $$(whoami)"
	@echo "PWD: $$(pwd)"
	@echo "Python executable: $$(which python)"
	@echo "Python version: $$(python --version)"
	@echo "Virtualenv active? $${VIRTUAL_ENV:-None}"
	@echo "Path variable: $$PATH"
	@if [ -f "./.venv/bin/activate" ]; then echo "Host .venv exists"; fi
	@if [ -d "/opt/venvs/kitchenwatch" ]; then echo "Container venv exists"; fi
	@if [ -f /.dockerenv ] || grep -qE 'docker|containerd' /proc/1/cgroup; then \
	    echo "🧱 Running inside a container"; \
	else \
	    echo "💻 Running on host machine"; \
	fi
	@if [ -d "$(VENV)" ]; then echo "Host $(VENV) exists"; else echo "Host $(VENV) missing"; fi
	@echo "======================="


# --- Code Quality --------------------------------------------------

## Run Ruff linter
lint:
	$(UV_RUN) ruff check --target-version py312 $(SRC_DIR) $(TEST_DIR)


## Run Ruff auto-fix and Black formatter
format:
	$(UV_RUN) ruff check --target-version py312 --fix $(SRC_DIR) $(TEST_DIR)
	$(UV_RUN) black $(SRC_DIR) $(TEST_DIR)


## Run Mypy type checker
typecheck:
	$(UV_RUN) mypy --python-version $(PYTHON_VERSION) $(SRC_DIR)

## Run full static analysis (Ruff + Mypy + Safety)
check: lint typecheck safety

## Check for known dependency vulnerabilities
safety:
	$(UV_RUN) safety check || true

# --- Testing -------------------------------------------------------

## Run tests with coverage
test:
	$(UV_RUN) pytest -v --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

## Run only async tests
test-async:
	$(UV_RUN) pytest -v -m "asyncio" $(TEST_DIR)

# --- Commit & Docs -------------------------------------------------

## Run all pre-commit hooks
precommit:
	$(UV_RUN) pre-commit run --all-files

## Check commit messages against Conventional Commits
cz-check:
	$(UV_RUN) cz check --rev-range origin/main..HEAD

# --- Containers ----------------------------------------------------

## Build dev container image
devcontainer-build:
	docker build -f .devcontainer/Dockerfile.dev -t kitchenwatch-dev ..

## Start dev container shell
devcontainer-shell:
	docker run --rm -it -v $(PWD):/workspace/kitchenwatch kitchenwatch-dev /bin/bash

# --- Python Cache ---------------------------------------------------------

## Clean all __pycache__ folders and recompile for current uv Python
pycache-clean:
	@echo "Cleaning all __pycache__ folders..."
	@find $(SRC_DIR) $(TEST_DIR) -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Recompiling Python files for $(PYTHON_VERSION)..."
	@uv run python -m compileall $(SRC_DIR) $(TEST_DIR) > /dev/null
	@echo "Done."




# --- Utility -------------------------------------------------------

## Show available make commands
help:
	@echo "Available commands:"
	@echo "----------------------------"
	@echo "  venv                Create or update the local virtual environment"
	@echo "  reinstall           Reinstall all dev dependencies cleanly"
	@echo "  env-info            Print environment info"
	@echo "  lint                Run Ruff linter"
	@echo "  format              Run Ruff auto-fix and Black formatter"
	@echo "  typecheck           Run Mypy type checker"
	@echo "  check               Run full static analysis (Ruff + Mypy + Safety)"
	@echo "  safety              Check for known dependency vulnerabilities"
	@echo "  test                Run tests with coverage"
	@echo "  test-async          Run only async tests"
	@echo "  precommit           Run all pre-commit hooks"
	@echo "  cz-check            Check commit messages against Conventional Commits"
	@echo "  devcontainer-build  Build dev container image"
	@echo "  pycahe-clean  		 Clean all __pycache__ folders and recompile for current uv Python"
	@echo "  help                Show available make commands"

# --- Phony Targets -------------------------------------------------

.PHONY: venv reinstall lint format typecheck check safety env-info \
        test test-async precommit cz-check devcontainer-build devcontainer-shell help

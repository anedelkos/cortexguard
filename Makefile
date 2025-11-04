.PHONY: venv reinstall env-info lint format typecheck safety check \
        test test-async test-unit test-integration test-e2e precommit cz-check \
        devcontainer-build devcontainer-shell pycache-clean \
        simulate-download simulate-fuse simulate-stream simulate-all demo-run \
        simulate-local simulator-build simulator-run-dev simulator-run-with simulator-clean \
        task-install task-help

# --- Setup ---
venv:
	@task venv

reinstall:
	@task reinstall

env-info:
	@task env-info

# --- Code Quality ---
lint:
	@task lint

format:
	@task format

typecheck:
	@task typecheck

safety:
	@task safety

check:
	@task check

# --- Testing ---
test:
	@task test

test-async:
	@task test-async

test-unit:
	@task test-unit

test-integration:
	@task test-integration

test-e2e:
	@task test-e2e

# --- Commit & Docs ---
precommit:
	@task precommit

cz-check:
	@task cz-check

# --- Containers ---
devcontainer-build:
	@task devcontainer-build

devcontainer-shell:
	@task devcontainer-shell

# --- Python Cache ---
pycache-clean:
	@task pycache-clean

# --- Simulator and demo ---
simulate-download:
	@task simulate:download

simulate-fuse:
	@task simulate:fuse

simulate-stream:
	@task simulate:stream

simulate-all:
	@task simulate:all

demo-run:
	@task demo:run

simulate-local:
	@task simulate:local

simulator-build:
	@task simulator:build

simulator-run-dev:
	@task simulator:run-dev

simulator-run-with:
	@task simulator:run-with

simulator-clean:
	@task simulator:clean

# --- Taskfile Bootstrap ---
task-install:
	curl -sL https://taskfile.dev/install.sh | sh
	mkdir -p ~/.local/bin
	mv ./bin/task ~/.local/bin/task
	@echo 'Task installed to ~/.local/bin/task'
	@echo 'Add this to your shell config: export PATH="$HOME/.local/bin:$PATH"'


help:
	@if ! command -v task >/dev/null; then \
        echo "❌ Task not found. Run 'make task-install' to install it."; \
        echo "Then add ~/.local/bin to your PATH."; \
        exit 1; \
    fi
	@task --list

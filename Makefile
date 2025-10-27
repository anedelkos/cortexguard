.PHONY: venv reinstall env-info lint format typecheck safety check \
        test test-async precommit cz-check \
        devcontainer-build devcontainer-shell pycache-clean \
        download fuse stream all demo \
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
download:
	@task simulate:download

fuse:
	@task simulate:fuse

stream:
	@task simulate:stream

all:
	@task simulate:all

demo:
	@task demo:run


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

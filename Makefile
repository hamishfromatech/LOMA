.PHONY: setup test lint doctor bench listen clean

setup:            ## create .venv, install deps, pull models
	./scripts/setup.sh

test:             ## run the test suite
	.venv/bin/python -m pytest

lint:             ## ruff
	.venv/bin/ruff check src tests

fmt:              ## auto-fix lint
	.venv/bin/ruff check --fix src tests

doctor:           ## end-to-end installation check
	.venv/bin/loma doctor

bench:            ## measure latency and RAM
	.venv/bin/loma bench
	.venv/bin/python scripts/bench_memory.py

listen:           ## talk to it
	.venv/bin/loma listen

tools:            ## list MCP tools
	.venv/bin/loma tools

clean:
	rm -rf .venv build dist *.egg-info .pytest_cache .ruff_cache

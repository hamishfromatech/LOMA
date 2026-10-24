.PHONY: setup test lint fmt doctor bench listen tools api web build-web clean

setup:            ## create .venv, install deps, pull models
	./scripts/setup.sh

test:             ## run the test suite
	.venv/bin/python -m pytest

lint:             ## ruff + eslint
	.venv/bin/ruff check src tests server
	cd web && npm run lint

fmt:              ## auto-fix lint
	.venv/bin/ruff check --fix src tests server

doctor:           ## end-to-end installation check
	.venv/bin/loma doctor

bench:            ## measure latency and RAM
	.venv/bin/loma bench
	.venv/bin/python scripts/bench_memory.py

listen:           ## talk to it
	.venv/bin/loma listen

tools:            ## list MCP tools
	.venv/bin/loma tools

api:              ## Flask API on :5050
	.venv/bin/python -m server.app

web:              ## Next.js dev server on :3000
	cd web && npm run dev

build-web:        ## production build of the UI
	cd web && npm run build

clean:
	rm -rf .venv build dist *.egg-info .pytest_cache .ruff_cache

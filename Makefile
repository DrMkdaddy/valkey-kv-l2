SHELL := bash
VALKEY_PORT ?= 6390
LLAMA_URL ?= http://127.0.0.1:8097
PY ?= python3

.PHONY: help test bench bench-restore chart check nix-bench nix-check

help: ## list targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-14s %s\n", $$1, $$2}'

test: ## lossless round-trip and tiering invariants (starts Valkey if needed)
	./scripts/with-valkey.sh $(PY) tests/test_tiered.py

bench: ## capacity benchmark, then render the chart from the committed sweep
	./scripts/with-valkey.sh bash -c '$(PY) tests/test_tiered.py && $(PY) bench/bench.py'
	$(PY) bench/chart.py

bench-restore: ## full crossover sweep against a live llama-server (rewrites results/)
	./scripts/with-valkey.sh $(PY) bench/sweep.py --url $(LLAMA_URL)
	$(PY) bench/chart.py

chart: ## re-render docs/restore-vs-prefill.svg from results/restore_sweep.json
	$(PY) bench/chart.py

check: ## run the tests and verify the committed chart is current
	./scripts/with-valkey.sh $(PY) tests/test_tiered.py
	$(PY) bench/chart.py --check

nix-bench: ## make bench inside the flake's dev shell
	nix develop --command make bench

nix-check: ## nix flake check (tests plus chart freshness)
	nix flake check --print-build-logs

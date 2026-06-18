# TA-MRC-PE-CC-Tube-MPC — task runner for Linux environments.
# Usage: make <target>

PYTHON := python
PYTEST  := $(PYTHON) -m pytest
PIP     := $(PYTHON) -m pip

.PHONY: help install test test-fast lint smoke aggregate figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install with CasADi solver backend
	$(PIP) install -e ".[solver]"

install-dev:  ## Install with all extras for development
	$(PIP) install -e ".[all]"

test:  ## Run full test suite
	$(PYTEST) tests/ -q

test-fast:  ## Run fast tests only (skip slow markers)
	$(PYTEST) tests/ -q -m "not slow"

lint:  ## Lint with ruff
	ruff check src/ tests/

smoke:  ## Run single-scenario smoke test
	$(PYTHON) scripts/run_single_scenario.py --scenario S2 --method Proposed --seed 1

aggregate:  ## Aggregate results (requires results/core/metrics_by_episode.csv)
	$(PYTHON) scripts/aggregate_results.py --input results/core --output results/core/aggregated

figures:  ## Generate paper figures (requires aggregated results)
	$(PYTHON) scripts/plot_paper_figures.py --input results/core --output results/core/figures

audit:  ## Audit failure cases
	$(PYTHON) scripts/audit_failure_cases.py --input results/core --output results/core/audit

stats:  ## Run statistical tests
	$(PYTHON) scripts/run_statistical_tests.py --input results/core --output results/core/analysis

dry-run:  ## Print experiment plan without running
	$(PYTHON) scripts/run_all_core.py --dry-run

quick-experiment:  ## Run quick smoke experiment
	$(PYTHON) scripts/run_all_core.py --quick --output results/quick

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/

.PHONY: install dev test lint forecast validate odds clean

install:          ## Install the package
	pip install -e .

dev:              ## Install with dev + gbm extras
	pip install -e ".[dev,gbm]"

test:             ## Run the test suite (offline)
	pytest

lint:             ## Lint with ruff
	ruff check src tests

forecast:         ## 2026 dual-track forecast (champion probs + match 1X2)
	wcforecast forecast

validate:         ## Walk-forward out-of-sample validation scorecard
	wcforecast validate

odds:             ## Live bookmaker consensus odds (needs ODDS_API_KEY)
	wcforecast odds

clean:            ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	rm -rf data/cache data/raw
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

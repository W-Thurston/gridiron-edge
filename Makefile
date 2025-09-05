.PHONY: install fmt lint typecheck test data-schedules backtest-smoke backtest clean

PY = poetry run
CLI = $(PY) gridiron

install:
	poetry install
	poetry run pre-commit install

fmt:
	poetry run isort .
	poetry run black .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src tests

test:
	poetry run pytest

# Usage: make data-schedules SEASONS=2010-2024
data-schedules:
	$(CLI) data-schedules --seasons "$(SEASONS)"

backtest-smoke:
	$(CLI) backtest --train-seasons 2010-2021 --predict-seasons 2022 --snapshot EARLY_WED_10ET --mode coinflip

# Usage: make backtest SEASONS=2015-2024 SNAPSHOT=T_MINUS_24H_ET MODE=home-bias
backtest:
	$(CLI) backtest --train-seasons $(SEASONS) --predict-seasons $(SEASONS) --snapshot $(SNAPSHOT) --mode $(MODE)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

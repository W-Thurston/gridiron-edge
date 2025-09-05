[![CI](https://github.com/<YOUR_ORG_OR_USER>/gridiron-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_ORG_OR_USER>/gridiron-edge/actions/workflows/ci.yml)
[![Release Please](https://github.com/<YOUR_ORG_OR_USER>/gridiron-edge/actions/workflows/release-please.yml/badge.svg)](https://github.com/<YOUR_ORG_OR_USER>/gridiron-edge/actions/workflows/release-please.yml)


# Gridiron Edge — Phase 0

Reproducible NFL prediction skeleton with leakage guardrails and walk-forward backtesting.

## Quick start
```bash
poetry install
poetry run pre-commit install

# 1) Ingest schedules (standardize to parquet)
make data-schedules SEASONS=2010-2024

# 2) Smoke backtest: train ≤ 2021, predict 2022 (moneyline dummy)
make backtest-smoke

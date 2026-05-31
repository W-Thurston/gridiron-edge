# Gridiron Edge

NFL game prediction, season simulation, and betting research platform.

Predicts game outcomes using Elo ratings, EPA features, and tree-based models. Simulates full seasons and playoff brackets via Monte Carlo. Tracks predictions against closing lines for CLV analysis.

## Quick start

```bash
uv sync
uv run gridiron --help
```

## Weekly workflow (during season)

```bash
# Refresh data + rebuild features
uv run gridiron run-data-pipeline

# Generate predictions and rankings
uv run gridiron output predictions --year 2026-2027 --week 1
uv run gridiron output ranks --year 2026-2027 --week 1

# Update playoff probabilities
uv run gridiron sim run
```

## Full bootstrap (new machine or season reset)

```bash
uv run gridiron run-data-pipeline \
  --all-years \
  --upcoming-season 2026 \
  --fit-elo-all-years \
  --season-year 2025-2026 \
  --skip fetch-odds
```

## Code quality

```bash
# Normal dev loop
uv run ruff check . --fix && uvx pyrefly check && uv run pytest -m "unit and not slow"

# Pre-commit handles: ruff, pyrefly, unit tests
# Pre-push handles: integration + e2e tests
```


### Testing

```bash
# Unit tests only (fast, your normal loop)
uv run pytest-m "unit and not slow"

# Integration + E2E (runs automatically on git push)
uv run pytest -m "integration or e2e"

# Slow tests
uv run pytest -m slow

# Full suite
uv run pytest

# With coverage
uv run pytest --cov --cov-report=term-missing
```
Tests follow a three-tier pyramid (unit → integration → e2e) with pytest markers.
Pre-commit hooks run unit tests on every commit; pre-push runs integration + e2e.

### Documentation
- [HANDOFF.md](HANDOFF.md) — how everything works, architecture, operational reference
- [PLAN.md](PLAN.md) — active roadmap and backlog
- [CHANGELOG.md](CHANGELOG.md) — what has been built and when
- [FEATURES.md](FEATURES.md) — comprehensive feature catalog across all domains
- [ROADMAP.md](ROADMAP.md) — long-term strategic direction

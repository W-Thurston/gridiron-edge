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
uv run ruff check src/ --fix
uvx pyrefly check
uv run pytest
```

## Documentation

- [HANDOFF.md](HANDOFF.md) — how everything works, architecture, operational reference
- [PLAN.md](PLAN.md) — active roadmap and backlog
- [CHANGELOG.md](CHANGELOG.md) — what has been built and when
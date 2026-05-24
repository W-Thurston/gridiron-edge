# NFL_Predict / Gridiron Edge

Predict NFL game outcomes using Pro Football Reference data and an Elo-based workflow.

## Quick start

```bash
uv sync
uv run gridiron --help
```

### Weekly data refresh

```bash
uv run gridiron run-data-pipeline --year 2025 --no-fetch-odds
```

### Elo outputs (Excel)

```bash
uv run gridiron ratings elo fit
uv run gridiron ratings elo predict --year 2025-2026 --week 16
uv run gridiron output ranks --year 2025-2026 --week 15
```

Weather (requires `OWM_API_KEY`):

```bash
export OWM_API_KEY="YOUR_KEY"
uv run gridiron ingest weather --season-year "2025-2026"
```

## Documentation

- [HANDOFF.md](HANDOFF.md) — architecture, data files, troubleshooting
- [PLAN.md](PLAN.md) — refactor checklist and validation commands

## Docker

```bash
docker build -t nfl_predict .
```

## Tests

```bash
uv run pytest
```
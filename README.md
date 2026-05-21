# NFL_Predict / Gridiron Edge

Predict NFL game outcomes using Pro Football Reference data and an Elo-based workflow.

## Quick start (recommended)

```powershell
poetry install
poetry run gridiron --help
```

### Weekly data refresh

```powershell
poetry run gridiron run-data-pipeline --year 2025 --no-fetch-odds
```

### Elo outputs (Excel)

```powershell
poetry run gridiron ratings elo fit
poetry run gridiron ratings elo predict --year 2025-2026 --week 16
poetry run gridiron output ranks --year 2025-2026 --week 15
```

Weather (requires `OWM_API_KEY`):

```powershell
$env:OWM_API_KEY="YOUR_KEY"
poetry run gridiron ingest weather --season-year "2025-2026"
```

## Documentation

- [HANDOFF.md](HANDOFF.md) — architecture, data files, troubleshooting
- [PLAN.md](PLAN.md) — refactor checklist and validation commands

## Deprecated root scripts

`PFR_data_pipeline_run.py` and `PFR_model_pipeline_run.py` print a deprecation notice. Use `poetry run gridiron` instead. `PFR_model_pipeline_run.py` still maps its flags to the new CLI for a transition period.

## Docker

Generate dependencies from Poetry, then build:

```powershell
poetry export -f requirements.txt --without-hashes -o requirements.txt
docker build -t nfl_predict .
```

## Tests

```powershell
poetry run pytest
```

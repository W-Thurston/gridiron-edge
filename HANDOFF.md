# Project Handoff: `NFL_Predict` / Gridiron Edge

How to run the project, how it is laid out, and what commonly breaks.

---

## What this project does

- **Goal**: NFL game prediction support — win probabilities, Elo rankings, weekly matchup visualisations, season simulation, and odds tracking.
- **Data source**: [nflverse](https://github.com/nflverse/nflverse-data) via `nfl_data_py` (replaces PFR/Scrapy).
- **Entry point**: `uv run gridiron` → [`src/gridiron_edge/cli.py`](src/gridiron_edge/cli.py)

---

## Repository layout

| Path | Role |
|------|------|
| [`src/gridiron_edge/`](src/gridiron_edge/) | Main package |
| [`src/gridiron_edge/ingest/nflverse/`](src/gridiron_edge/ingest/nflverse/) | nflverse game + schedule ingestion |
| [`src/gridiron_edge/ingest/pfr/collector_impl.py`](src/gridiron_edge/ingest/pfr/collector_impl.py) | Weather (OWM) + DraftKings odds |
| [`src/gridiron_edge/ingest/odds/`](src/gridiron_edge/ingest/odds/) | Odds ledger (Parquet, long format) |
| [`src/gridiron_edge/transform/clean/`](src/gridiron_edge/transform/clean/) | nflverse → canonical schema mappers |
| [`src/gridiron_edge/features/`](src/gridiron_edge/features/) | Feature registry + pipeline |
| [`src/gridiron_edge/ratings/elo/`](src/gridiron_edge/ratings/elo/) | Elo table, fit, predict, evaluate |
| [`src/gridiron_edge/sim/`](src/gridiron_edge/sim/) | Monte Carlo season + playoff simulation |
| [`src/gridiron_edge/viz/`](src/gridiron_edge/viz/) | Predictions image/HTML, playoff table, rankings CSV |
| [`src/gridiron_edge/core/`](src/gridiron_edge/core/) | Settings, logging, console output |
| [`data/`](data/) | Generated at runtime (not committed) |

**Data layout:**

```
data/
  raw/          nflverse Parquet files (games, upcoming schedule)
  cleaned/      Canonical CSVs (games, schedule, Elo state, stadiums)
  modeling/     Feature matrix (base + full)
  output/
    predictions/{year}/   week_NN_predictions.png + .html + .csv
    rankings/             elo_rankings_{year}_wkNN.csv
    sim/                  playoff probability tables
  odds/
    dk_odds_log.parquet   Full historical ledger (all pulls)
    dk_odds_current.parquet  Latest pull snapshot (for viz)
```

---

## Setup

```bash
uv sync
uv run gridiron --help
```

**Python**: `>=3.12,<4`.

---

## Configuration

| Secret / setting | How |
|------------------|-----|
| OpenWeather | Env var `OWM_API_KEY` for `gridiron ingest weather` |
| Data paths | [`datasets/registry.py`](src/gridiron_edge/datasets/registry.py) + [`core/settings.py`](src/gridiron_edge/core/settings.py) |

---

## Primary workflows

### Full bootstrap (first run or season reset)

```bash
uv run gridiron run-data-pipeline \
  --all-years \
  --upcoming-season 2026 \
  --build-elo \
  --fit-elo-all-years
```

Fetches all history (1999–present), 2026 upcoming schedule, rebuilds Elo, builds feature matrix. ~115s.

### Weekly refresh (during season)

```bash
# No flags needed — auto-detects current season
uv run gridiron run-data-pipeline
```

Re-fetches current season games + upcoming schedule, refreshes Elo incrementally, rebuilds features.

### Specific season

```bash
uv run gridiron run-data-pipeline --season 2025
```

### Step-by-step

```bash
uv run gridiron ingest nflverse-games              # current season
uv run gridiron ingest nflverse-games --all-years  # full history
uv run gridiron ingest nflverse-upcoming           # current season upcoming
uv run gridiron transform clean-games
uv run gridiron transform clean-upcoming
uv run gridiron ratings elo fit [--all-years]
uv run gridiron features model-inputs
```

### Output commands

```bash
# Weekly matchup predictions (PNG + HTML)
uv run gridiron output predictions --year 2026-2027 --week 1

# Elo rankings CSV
uv run gridiron output ranks --year 2026-2027 --week 1

# Season simulation
uv run gridiron sim run [--n-sims 10000]
```

### DK odds

```bash
# Pull current week odds → appends to ledger + writes snapshot
uv run gridiron ingest dk-odds --season 2026-2027 --week 1
```

### Weather

```bash
export OWM_API_KEY="YOUR_KEY"
uv run gridiron ingest weather --season-year "2026-2027"
```

---

## File contract (key artifacts)

| File | Purpose |
|------|---------|
| `data/raw/NFL_wk_by_wk_nflverse.parquet` | Raw nflverse games (all seasons) |
| `data/cleaned/NFL_wk_by_wk_cleaned.csv` | Canonical historical games |
| `data/raw/NFL_upcoming_schedule_nflverse.parquet` | Raw upcoming schedule |
| `data/cleaned/NFL_upcoming_schedule_cleaned.csv` | Canonical upcoming schedule |
| `data/cleaned/NFL_Team_Elo.csv` | Elo ratings state table |
| `data/modeling/base_modeling_file.csv` | Base modeling rows |
| `data/modeling/modeling_file.csv` | Full feature matrix |
| `data/cleaned/NFL_stadium_reference.csv` | Stadium / geo reference |
| `data/odds/dk_odds_log.parquet` | Full DK odds history (long format) |
| `data/odds/dk_odds_current.parquet` | Latest DK odds snapshot for viz |

---

## Where to read code

- CLI: [`src/gridiron_edge/cli.py`](src/gridiron_edge/cli.py)
- nflverse ingest: [`src/gridiron_edge/ingest/nflverse/`](src/gridiron_edge/ingest/nflverse/)
- Weather + DK odds collector: [`src/gridiron_edge/ingest/pfr/collector_impl.py`](src/gridiron_edge/ingest/pfr/collector_impl.py)
- Odds ledger: [`src/gridiron_edge/ingest/odds/store.py`](src/gridiron_edge/ingest/odds/store.py)
- Features: [`src/gridiron_edge/features/pipeline.py`](src/gridiron_edge/features/pipeline.py)
- Elo: [`src/gridiron_edge/ratings/elo/`](src/gridiron_edge/ratings/elo/)
- Simulation: [`src/gridiron_edge/sim/`](src/gridiron_edge/sim/)
- Predictions viz: [`src/gridiron_edge/viz/predictions.py`](src/gridiron_edge/viz/predictions.py)
- Console output: [`src/gridiron_edge/core/console.py`](src/gridiron_edge/core/console.py)

---

## Code quality

```bash
uv run ruff check src/ --fix   # lint
uv run ruff format src/        # format
uvx pyrefly check              # type check
uv run pytest                  # tests (14/14)
```

All four must pass before committing. Config in [`pyproject.toml`](pyproject.toml) (Ruff) and [`pyrefly.toml`](pyrefly.toml) (Pyrefly).

Use `uv run gridiron -v <command>` for verbose output with step detail, row counts, and debug logs. `GRIDIRON_VERBOSE=1` also works.

---

## Known sharp edges

### nflverse data cadence

nflverse updates nightly after each game day. The cleanest weekly snapshot is Thursday (after the NFL incorporates Mon–Wed stat corrections). Historical coverage goes back to 1999.

### DK odds `--week` default

`fetch_dk_odds()` defaults `week=1` if not provided. Always pass `--week` explicitly when pulling mid-season to ensure the ledger is tagged correctly.

### Elo and upcoming week

`output predictions` merges Elo onto the upcoming schedule. If the Elo state table doesn't yet have entries for the requested week, predictions will be empty. Run `ratings elo fit` first.

### `collector_impl.py` scope

`collector_impl.py` now handles only weather (OpenWeatherMap) and DraftKings odds. All historical game + schedule ingestion goes through `ingest/nflverse/`. Do not add Scrapy or PFR dependencies back — that path was retired.

---

## Operational checklist (weekly)

1. `uv run gridiron run-data-pipeline` — refresh games + upcoming + features
2. `uv run gridiron ingest dk-odds --season YYYY-YYYY+1 --week N` — pull odds
3. `uv run gridiron output predictions --year YYYY-YYYY+1 --week N` — generate viz
4. `uv run gridiron sim run` — update playoff probabilities
5. `uv run gridiron output ranks --year YYYY-YYYY+1 --week N` — rankings CSV

---

## Further work

See [`PLAN.md`](PLAN.md) for full phase history and next priorities.
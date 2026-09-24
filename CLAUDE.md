# Gridiron Edge

File-backed NFL decision-support platform: Python CLI (`uv run gridiron`),
persisted model/evaluation artifacts, a mostly read-only FastAPI service,
and a generated-contract React frontend in `frontend/`.

## Read first
- Before planning or editing anything, read "Ways of Working" in PLAN.md. It is binding.
- Doc roles: PLAN.md = the one active unit; ROADMAP.md = future scope;
  HANDOFF.md = how the system works today; DECISIONS.md = locked architecture;
  CHANGELOG.md = what shipped.
- HANDOFF.md is ~1,200 lines. Don't read it top to bottom. Grep its headings
  and read only the section that owns the boundary you're touching.
- Inspect the repo before assuming any file, command, schema, or test exists.

## Commands
- Setup: `uv sync`; `cd frontend && pnpm install`
- Run locally: `uv run gridiron api serve` + `cd frontend && pnpm dev`
- Python gate: `uv run ruff check . --fix && uv run ruff format . && uvx pyrefly check && uv run pytest -m "unit and not slow"`
- Frontend gate: `cd frontend && pnpm lint && pnpm build && pnpm test:run`
- After any API contract change: `uv run gridiron api export-schema`, then
  `cd frontend && pnpm gen:api`, then the frontend gate.

## Invariants (breaking these is a bug, even if tests pass)
- One canonical Away/Home row per game. Win models predict `HOME_WIN`;
  differentials are Home minus Away. Never reintroduce `TEAM_A`/`TEAM_B`,
  `HOME_FIELD`, doubled team rows, or `RESULT` as a target.
- API requests only serialize persisted state: no model inference, no
  "latest forecast" by recency, no Elo fallback, no request-time generation of
  products, edges, policy, or Kelly sizing. `POST /portfolio/bets` is the only write route.
- Never hand-edit `api-schema.json` or `frontend/src/api/schema.ts`. Regenerate them.
- Immutable weekly products and persisted evidence are never modified in place.
- Prediction readiness and market readiness are independent.
- Frontend never silently hides unavailable data; use the shared field-status components.
- No backward compatibility for development-era contracts unless a current contract requires it.

## Ask before running
- Anything that calls external sources: The Odds API, nflverse, weather backfill.
- Full retrain, historical backfills, worker deployment, and slow/integration/e2e suites.
- Anything that writes to `data/` outside an owning command, or records a wager.

## Commits
- One focused commit per unit: Conventional Commit subject, bullet-list body
  (implementation, tests, docs). The PLAN.md closeout goes in the same commit.
- Follow /closeout for closeout. Don't commit or push without my OK.
- Pre-commit runs lint, types, and unit tests; pre-push adds integration and e2e.
  Run the gate yourself first so hooks don't fail mid-commit.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for W-Thurston/gridiron-edge, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

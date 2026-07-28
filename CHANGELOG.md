# Gridiron Edge - Changelog

What has been built and when. Newest first.

---
## 2026-07-28 — PlayoffProjections navigation and Weekly Outcomes

Extended the PlayoffProjections rebuild with explicit navigation from Team
Rankings and a league-wide weekly schedule-probability matrix.

### Navigation

- Added a shared Team Rankings / Playoff Projections sibling switcher.
- Added the switcher above both screens with active-page semantics and route
  coverage.
- Preserved `/teams?team={abbr}` navigation from both projections views.

### Weekly Outcomes API

- Added `GET /projections/grid`.
- Added frozen Pydantic schemas for the response, team rows, and weekly states.
- Added a static source container and loader for:
  - `data/output/temp/season_grid.csv`;
  - the cleaned upcoming schedule;
  - completed regular-season games;
  - unified long-name / abbreviation mappings.
- Added hand-written serialization for played, projected, bye, and unavailable
  states.
- Added opponent, home/away perspective, game ID, date, time, weekly win
  probability, actual W/L/T result, and `completed_through_week`.
- Determined byes from schedule membership rather than treating a
  `Wxx_WIN_P == 0` artifact value as a bye or loss.
- Added `no_schedule_data` field-status metadata.
- Regenerated `api-schema.json` and `frontend/src/api/schema.ts`.

### Weekly Outcomes frontend

- Added the `useProjectionGrid` React Query hook.
- Added `WinProbabilityCell`, a full-table-cell primitive using a fixed
  diverging red-neutral-green scale centered at 50%.
- Added Playoff Chances / Weekly Outcomes local views.
- Added Week 1–18 rows with:
  - grouped Played Games / Projected Games headers;
  - a clear played/projected boundary;
  - explicit BYE cells;
  - a sticky Team column;
  - shared conference and dependent division filters.
- Preserved filters while switching local views.
- Reused one team-identity implementation across Playoff Chances and Weekly
  Outcomes.

### Matchup details and accessibility

- Added pointer-hover and keyboard-focus matchup details.
- Portaled tooltips to `document.body` so the horizontal-scroll container does
  not clip them.
- Added viewport clamping at top, left, and right edges.
- Added responsive tooltip width for long team names.
- Formatted tooltips as three centered rows:
  - team matchup;
  - week, date, and time;
  - projected chance to win or played result.
- Kept numeric percentages, explicit BYE labels, grouped headers, and
  accessible names as non-color encodings.

### Verification

- Verified the real response contains 32 teams and 18 weekly entries per team
  for the 2026–2027 preseason artifact.
- Verified `completed_through_week = 0` in the preseason state.
- Verified scheduled games include opponent, venue perspective, date, time,
  and probability.
- Verified Arizona Week 14 is a confirmed BYE with null probability rather
  than a zero-percent game.
- Verified conference and division filtering, local-view state persistence,
  played/projected grouping, sticky team identity, tooltip edge behavior,
  keyboard access, narrow-width scrolling, and long matchup names.
- Backend quality gates, frontend build, focused tests, and the full frontend
  test suite pass.

### Deferred

- Probability-cell texture remains deferred pending color-vision review.

## 2026-07-28 — PlayoffProjections rebuild

Rebuilt the playoff-projections screen as a live, interactive counterpart to
the original static playoff table.

### API contract

- Renamed `week_over_week_delta` to the semantically explicit `elo_delta`.
- Added `items.elo_delta = no_prior_snapshot` status metadata when no usable
  prior same-season Elo snapshot exists.
- Preserved partial-null behavior without incorrectly marking the entire field
  unavailable.
- Regenerated `api-schema.json` and the TypeScript API schema.

### Frontend

- Added `HeatCell`, using a fixed absolute 0–1 probability scale and full-cell
  heat coloring.
- Added accessible `SortableHeader` with explicit active direction.
- Added sortable columns for team, current Elo, Elo delta, average wins, and
  every postseason probability stage.
- Added dependent conference and division selectors.
- Composed current Elo, current record, conference, division, colors, and
  as-of week from the existing `/teams` response.
- Added season, as-of-week, simulation-count, and computed-time context.
- Replaced repeated Week 1 Elo warnings with quiet row placeholders and one
  explanatory legend caveat.
- Preserved warnings for unexpected missing Elo deltas after Week 1.
- Added team navigation to `/teams?team={abbr}`.
- Preserved all postseason stages at narrow widths through horizontal overflow.

### Verification

- Verified all 32 teams against the real Week 1 projections artifact.
- Verified 16-team conference filters and four-team division filters.
- Verified sorting, filter resets, team navigation, continuous heat treatment,
  Week 1 Elo handling, narrow-width behavior, and highlight mode.
- Frontend build and test suite pass.
- Targeted projections backend tests, Ruff checks, and Pyrefly pass.


## 2026-07-01 — W9.10 Compare Screen Rebuild

Two-mode matchup surface at `/compare`. Both modes prototype-aligned
against real, pipeline-populated data.

### Team vs Team

- Mode switcher (Pill, URL-synced `?mode=`)
- Mirrored team pickers (name→logo / logo→name, inward-facing) + swap
  button, width-constrained + centered
- Cohort strip (Season/L4/Home/Away)
- Separate cards: floating pickers → narrative → collapsible summary
  (centered team-left/right) → three matchup cards
- Matchup rows: 5-column center-aligned, mirrored rank-fill bars
  (rank 1 = full, filling toward center), edge chips (arrow + team +
  descriptor), value+rank inline, descriptive sublabels, title-style
  metric names
- Auto-narrative card (biggest collision per direction from rank diffs)
- 11-metric cohort_splits expansion: added def_pass_epa,
  def_third_down_pct, def_redzone_td_pct so every offensive metric has
  its reciprocal defensive-allowed pair

### Player vs Defense

- Independent player / stat-category / team selection (retired the
  prop_id model)
- Searchable player combobox (client-filtered `/players` roster)
- Stat category derived from player position (each option carries B1
  statKey + B3 stat_type)
- Team dropdown (all 32, independent)
- 7-split strip: Season/L4/Home/Away live + vs-Winning/Losing/Top-10
  pending (non-clickable, highlight-marked)
- Per-game bar chart (new BarChart primitive): player's stat as bars
  (B1) + team split-average as a solid reference line (B3) that moves
  with the split while bars stay static
- "Matchup, plainly" verdict card: big avg-allowed, rank-as-context
  line, baseline-driven verdict (defense-allowed vs player's own
  average → Favorable/Tough/Neutral, lean over/under), quantified delta
- By-split comparison table: player avg (from bars, per split) vs
  defense-allowed + Def rank across 4 live splits; 3 pending rows

### Backend (Path C — built to unblock the frontend)

- **B1** — `GET /players/{id}/history?stat=&season=&limit=`: per-game
  stat series from player_game_logs. **Also fixed a root-cause game_id
  scramble** — `_join_game_id` assigned merge-result Series onto a
  non-contiguous index (upstream dropna), scrambling game_id to same-
  week neighbors. Fix: reset_index before the 1:1 merges; derive
  trustworthy is_home. Regenerated logs; re-ran props compute-splits
  (had aggregated against wrong game contexts).
- **B2** — opponent_allowed expanded {season, l5} → {season, l4, home,
  away}. Home/away is the DEFENSE's perspective (inverse of offensive
  player is_home).
- **B3** — `GET /defense/{team}/allowed?stat_type=`: per-team allowed
  aggregates (all cohorts), keyed on arbitrary team (independent-team
  picker needs this; /compare/player is prop-keyed).
- **B4** — `GET /players?season=`: skill-player roster for the picker,
  deduped to latest team.

### New primitives

- **BarChart** — SVG bars + Y-grid + solid horizontal reference line
  with value tag. Reusable (PlayerProp game-log chart is a future
  consumer).

DistributionChart retired from Compare (BarChart replaced it); still
used by PlayerProp.

### Prototype-alignment adjustments (Team mode)

Six post-build refinements per side-by-side review: keep cards (not
floating), mirrored inward team icons, narrative as separate card,
collapsible summary card, centered ranking-bar comparison rows, three
separate matchup cards. Plus: value+rank inline, descriptive sublabels,
title-style center metric names, shortened fill bars, centered page
(max-width), grid centering within cards.

### Deferred

- Book line + O/U bar coloring — odds (W7); legend PendingChip, no
  fake line
- vs-winning / vs-losing / vs-top-10 splits — 3 pending pills + table
  rows (medium backend: opponent-record + self-ranking)
- Change 6 (sortable matchup rows by category/edge + drag-reorder) —
  P2, §9.8; build only if missed

### Substep arc

Tier 1 (mode switcher) → Tier 2 (Team vs Team + 6 alignment
adjustments) → backend B1–B4 (+ game_id fix) → C1 (pickers/strip) →
C2 (bar chart) → C3 (verdict + table) → C4 (cleanup + close-out).

## 2026-07-07 — W9.9 PlayerProp Rebuild

Rebuilt PlayerProp screen (`/players/:propId`) from skeleton with 6
ComingSoonCards to prototype fidelity across 8 substeps in 4 tiers.
Consumes existing data from Step 5 (situational splits) + Step 6
(player vs defense) + game + team metadata endpoints. New
DistributionChart primitive extractable to Compare screen (W9.10).

### Shipped

**Tier 1 — Layout restructure (1 substep):**
- Rebuild GameDetail-style skeleton: full-width hero placeholder +
  single-column content below
- Preserve existing content in placeholder cards (`HeroPlaceholder`,
  `DistributionPlaceholder`, `SectionPlaceholder`)
- Player vs Defense table unchanged
- Blocked ComingSoonCards reduced from 6 to 5 (Situational Splits
  removed from grid — will be real card in Tier 3b)

**Tier 2 — Hero header (2 substeps):**
- 2a: Player hero band with team-colored gradient (180deg, 30% mix →
  var(--bg-1)), TeamMark 56px, breadcrumb, big serif player name
- 2b: Prop summary callout — distinct card with green accent border,
  stat label with game context "MON vs SF", big em-dash for pending
  line, model mean + range on flex row, pending markers for
  confidence + EV
- "+ Bet slip" button outside the summary card

**Tier 3 — Content sections (3 substeps):**
- 3a: `DistributionChart` primitive at `components/primitives/`. SVG
  Gaussian density curve with 90% credible band shading, dashed
  vertical mean marker with value label, x-axis endpoints for
  lo_90/hi_90. Line marker slot (unused; pending). Responsive width
  via viewBox.
- 3b: `SituationalSplitsCard` consuming Step 5 `situational_splits`
  field. 8 cohorts in canonical order (Season / Last 4 / Home /
  Away / Favored / Underdog / Indoor / Outdoor). Format: "X.X avg ·
  N games". Empty state when field null (many props currently
  pending).
- 3c: Player vs Defense polish — WhyLink dot in header (info tone,
  kind: prop_defense), table headers restyled with uppercase small
  letter-spacing to match card language.

**Tier 4 — Placeholders + cleanup (2 substeps):**
- 4a: Change ComingSoonCards grid from 2-col to 3-col (`repeat(3, 1fr)`)
  for better use of horizontal space (5 cards fit as 3 + 2 rows).
- 4b: Cleanup verification — no dead code, no unused imports.

### Architecture consumed

**All 5 W9.5 primitives:**
- `TeamMark` — hero band + prop summary
- `Pill` — not used in PlayerProp; deferred
- `WhyLink` — dot variant on Player vs Defense card
- `Spark` — not used; new dedicated `DistributionChart` primitive
- `TeamHero` — not composed inline; hero band custom-built

### New primitive

**DistributionChart** — at `components/primitives/DistributionChart.tsx`.
- Renders Gaussian PDF from mean + std
- 90% credible band shading (filled area between lo/hi)
- Dashed vertical marker at mean + value label above
- Endpoint labels for lo_90 + hi_90
- Line marker slot (currently unused; awaits odds data)
- Fallback: renders "No distribution data available" when mean or std
  unavailable
- Responsive width via SVG viewBox

Will pay dividends in Compare screen (W9.10) Player vs Defense mode.

### New helper

**`utils/props.ts`** — extracted `formatStatType` and
`formatStatTypeShort` helpers. Same slug → display mapping used across
Dashboard PropEdgesRail, GameDetail TopPropEdgesCard, and PlayerProp.
Dashboard and GameDetail migrations from inline helper deferred to
follow-up cleanup.

### Backend gaps that surfaced

Same as previously identified in ROADMAP §9.7:
- Line context data (line, lean, confidence_tier, p_over) — blocked
  on odds join
- Situational splits data pending for many props
- Player game history endpoint — not consumed yet in W9.9
- Related props filter endpoint — not consumed yet

### Design tension noted

Prototype's PlayerProp had a right-side rail with "Why the model
leans", line shopping mini-table, and related props sidebar. Our
narrower app width doesn't accommodate a rail without cramping.
Consistent with W9.7 lesson — same design constraint applies.
Deferred right-rail elements to blocked ComingSoonCards on our
main content flow.

### Test coverage

Existing 60 tests continue passing. No new tests added for PlayerProp
components — coverage would be integration-level. Primitive tests
from W9.5 provide indirect coverage of building blocks.

### What's not shipped

- No player game history chart (backend endpoint blocked, §9.7)
- No line shopping mini-table (W7 blocked)
- No related props sidebar (backend filter blocked, §9.7)
- No "Why the model leans" reasoning (blocked on feature attribution)

### Next

Between workstreams. Options: W9.8 (backend enablers batch),
W9.10 (Compare rebuild — largest remaining screen gap), or another
polish sweep.

## 2026-07-07 — W9.7 Teams Split-View Rebuild

Restructured `/teams` and `/teams/:abbr` from two separate screens
into a single split-view at `/teams` with optional `?team=X` param.
Left column shows the league rankings; right column shows the
selected team's profile. Clicking a rankings row updates the right
pane without navigation, preserving ranking context across team
browsing.

### Shipped

**Tier 1 — Route restructure (1 substep):**
- Consolidated `TeamRankings` and `TeamProfile` into single
  `TeamsScreen.tsx`
- Auto-select #1 team silently when no `?team=` param
- Row click updates URL param via `navigate("/teams", {team: abbr})`
- Router routes /teams to TeamsScreen (both with and without team param)

**Tier 2 — Left column (2 substeps):**
- Enhanced rankings table with hover state on rows and selection
  highlight
- Trend column with signed colored pill (green/red/dim)
- 5-tab strip (Overall / Offense / Defense / ATS / Net Rating)
  via Pill primitive
- Overall (default) renders 32 teams sorted by rating
- Offense / Defense / ATS / Net Rating render `BlockedTabState`
  with §9.7 backend reference

**Tier 3 — Right column sections (4 substeps):**
- **Team hero band** — team-colored vertical gradient
  (180deg, 30% mix top → var(--bg-1) bottom). TeamMark (56px) left,
  breadcrumb (conf/div · rank · season · through week) above serif
  italic team name, inline hero stats (Record/Rank/Rating)
- **Rating chart** — new `RatingChart` primitive with Y-axis grid +
  rating labels, dots at each data point, X-axis week labels every
  ~4-5 weeks, inline W/L markers per week (green W below line, red
  L above line)
- **Situational Splits card** — Pill cohort switcher (Season/L4/Home/
  Away), 8 metrics (off/def EPA, breakdowns, situational percentages,
  turnover diff) from `cohort_splits` field
- **Recent Results** — existing `RecentResultsStrip` (unchanged)
- **Postseason Outlook** — composed from `/projections`, 5 rows
  (Make Playoffs, Reach Divisional, Reach Conf. Championship, Reach
  Super Bowl, Win Super Bowl) with colored progress bars mapping
  probability to fill width

**Tier 4 — Placeholders + cleanup (2 substeps):**
- Schedule Difficulty placeholder (blocker: `schedule_difficulty`,
  roadmap `§9.7`)
- Top Players placeholder (blocked)
- Deleted `TeamRankings.tsx` and `TeamProfile.tsx` files
- Deleted commented dead code (`ProfileCell`, `InlineFieldStatus`)

**Substep 4c — Polish sweep (7 adjustments in one substep):**
- Left column narrower (5fr / 11fr split); gap between columns
- Hero band aligned within profile column (removed negative margin
  after realizing our container width doesn't match prototype's
  full-screen assumption)
- Green 3px left border on selected rankings row
- Rankings subheader "Wk N · model v4.2"
- Single-column layout beneath hero (reverted from failed 80/20
  attempt; prototype's 80/20 works at ~1400px, ours at ~800px
  didn't fit)
- Postseason outlook: colored progress bars per row
- Rating chart W/L markers moved from X-axis text to inline
  (below line for W, above for L)

### Architecture consumed

**All 5 W9.5 primitives:**
- `TeamMark` — throughout (colored via cache)
- `Pill` — rankings tabs + cohort switcher
- `WhyLink` — not used in Teams; deferred
- `Spark` — not used; new dedicated `RatingChart` primitive instead
- `TeamHero` — not used inline; hero band composition doesn't fit
  its API. Composed directly in TeamsScreen.

### New primitive

**RatingChart** — SVG line chart with:
- Responsive width via viewBox
- Y-axis grid + rating value labels
- Line + data point dots
- X-axis labels every 4-5 weeks
- Optional recentResults prop for inline W/L markers

Location: `frontend/src/components/primitives/RatingChart.tsx`

### Helpers established

- `stripCityPrefix` — same pattern as GameDetail (backend returns
  full name, we strip city prefix to render "New England Patriots"
  as "New England _Patriots_" italic split)
- `expandDivisionLetter` — N/S/E/W → North/South/East/West
- `formatSeason` — "2025-2026" → "2025"

### Composition patterns

- Split-view route: single URL, optional query param drives selection
- Cross-endpoint composition: `/teams` + `/projections` joined
  client-side by team_abbr for postseason outlook
- Blocked-state tabs on `Pill` primitive: tabs remain clickable and
  show blocker messaging when selected, consistent with `field_status`
  pattern

### What surfaced

**Design tension:** Prototype uses ~1400px full-screen layout; our
app is centered ~800px. Two-column layouts that work in prototype
(e.g., 80/20 rating chart + narrow rail) don't fit our width. Reverted
to single-column below hero band. Documented for future workstreams —
prototype fidelity work will always need this constraint check.

### Test coverage

Existing 59 tests continue passing. No new tests added for TeamsScreen
components — coverage would be integration-level. Primitive tests from
W9.5 provide indirect coverage of building blocks.

### What's not shipped

Preserved as `ScheduleDifficultyPlaceholder` and Top Players
`ScaffoldCard`:
- Schedule Difficulty (blocker: `upcoming_games` backend enrichment)
- Top Players (blocker: WAR feature attribution)

Not consumed:
- WhyLink primitive (opportunity for future explainability affordance
  on rating chart or team stats)

### Backend gaps that surfaced

None new. Same gaps as previously identified in ROADMAP §9.7:
- `upcoming_games` enrichment for Schedule Difficulty
- WAR data for Top Players
- Off/def rating decomposition for Offense/Defense/Net Rating tabs
- Cumulative ATS record for ATS tab
- Enriched RecentResult with spread/ATS/O-U (not consumed in W9.7)

### Next

Between workstreams. Options: W9.8 (backend enablers), W9.9
(PlayerProp rebuild), or something else.

## 2026-07-07 — W9.6 GameDetail Full Fidelity

Rebuilt GameDetail (`/games/:id`) from skeleton with 5 coming-soon
cards to prototype fidelity across 9 substeps in 4 tiers. Uses all
5 primitives shipped in W9.5.

### Shipped

**Tier 1 — Layout restructure (1 substep):**
- Full-width header slot + 2-column grid (3fr main / 2fr rail)
- Preserved existing prediction data in placeholder cards during transition
- All old flat layout removed

**Tier 2 — Header composition (2 substeps):**
- Team hero header: two TeamHero components (right-oriented away,
  left-oriented home) framing center block with kick date + "at" +
  venue/weather placeholders
- Model lean callout composed from `/edges` filtered to game_id:
  recommendation + EV% + confidence tier + WhyLink dot + slip button
- Empty state "No model edge" when no edges available for game

**Tier 3 — Main column cards (3 substeps):**
- **Lines & Model Fair Value table** — 3 rows × 3 columns:
  - Market row: em-dashes (blocked on W7)
  - Gridiron Edge fair row: spread + total + moneyline (probability
    → American via new `probToAmerican()` helper in utils/odds.ts)
  - Recommendation row: top edge per market from `/edges` filter;
    highlighted with green tint
- **Win Probability card** — 2 columns:
  - Left: two prob bands with team label + big % + range label
  - Right: projected score display + margin string
  - Away band derived from home band (1 - home_hi/lo)
- **Team Comparison card** — 8 metrics × 4 cohorts:
  - Season / Last 4 / Home / Away tabs via Pill primitive
  - Simple 3-column layout (away value / metric / home value)
  - Green + bold coloring on winning team per metric
  - "Open full comparison →" button navigates to Compare with team
    abbrevs prefilled
  - Consumes `team_comparison` field from Step 7c

**Tier 4 — Right rail + cleanup (3 substeps):**
- **Top Prop Edges card** — compact right-rail list:
  - Filters `/props` by game_id (client-side)
  - Sort by predicted_mean descending, take top 4
  - Each row: player + TeamMark + position + confidence tier +
    stat + lean + line + model value + WhyLink dot + slip button
  - Row click → PlayerProp; "See all N →" shows count of props
- Placeholder integration: Swing Factors + Injuries remain as
  ComingSoonCard (blocked on named workstreams)
- Cleanup: deleted SectionPlaceholder dead code

### Architecture consumed

- **All 5 W9.5 primitives:** TeamHero (heavy — 4 usages), Pill
  (Team Comparison cohort tabs), WhyLink (model lean + prop edges),
  TeamMark (throughout, colored via cache), Spark (not directly
  used in GameDetail — future win prob chart candidate)

### New helpers

- `stripCityPrefix()` — removes "Kansas City " prefix from
  "Kansas City Chiefs" when city and name are exposed separately.
  Otherwise displays "Kansas City Kansas City Chiefs".
- `probToAmerican()` in `utils/odds.ts` — win probability to
  American odds using standard formula: `prob >= 0.5` (favored)
  gives negative American; `prob < 0.5` gives positive American.
- `formatKickLabel()` — game_date → "SUN · FEB 8" (mono uppercase)
- `formatMargin()` — model_spread + team names → "TEAM by X.X"
- `formatSpreadDisplay()`, `formatTotalDisplay()`, `formatMLDisplay()`
  — two-line stacks per Lines table cell
- `formatStatType()` — "qb_pass_yards" → "Pass Yds" for prop rows

### Composition patterns established

- Header + right-side callout wrapped in outer flex container with
  space-between
- Composed cards from multiple endpoints (game, edges, props) via
  React Query
- Client-side filtering by game_id for cross-endpoint composition
  (no backend filter params needed)
- Bet slip integration via `useBetSlip.add()` with placeholder -110
  odds (real odds arrive with W7)

### Test coverage

Existing 59 tests continue passing. No new tests added for GameDetail
sub-components — coverage would be integration-level (real API data
in browser). Primitive tests from W9.5 provide indirect coverage of
building blocks.

### What's not shipped

Preserved as `ComingSoonCard` placeholders for future work:
- Swing Factors (blocked on feature attribution workstream)
- Injuries (blocked on §5.3 injury data source)

No `Spark` usage in GameDetail. Future win probability chart or
prop distribution overlay could use it.

### Backend gaps that surfaced

None new. Same gaps as previously identified in ROADMAP §9.7:
- `kick_time` not exposed (game_date only)
- `venue_text`, `weather_text` pending
- `market_spread`, `market_total`, moneyline market lines blocked on W7

### Next

Between workstreams. Options: W9.7 (Teams split-view), W9.8 (backend
frontend-enablers), W9.9 (PlayerProp rebuild), or another slice of
polish work.

## 2026-07-06 — W9.5 Frontend Polish (Dashboard Rebuild + Cross-Cutting Primitives)

Small workstream between W8 close-out and next major work. Focused on
two things: rebuild the Dashboard (unusable debug scaffolding from
W9 Tier 1) into a real landing page, and ship 5 cross-cutting shared
primitives identified in the prototype audit.

### Shipped

**Tier 1 — Backend patch (1 substep):**
- Added `city`, `conference`, `division`, `primary_color`, `secondary_color`
  fields to `TeamRankingRow` and `TeamProfile` schemas.
- Consolidated `NFL_long_to_short_name.csv` and `NFL_conference_division.csv`
  into unified `NFL_team_metadata.csv`.
- Registry migrations: `teams_long_short` + `divisions` → `team_metadata`.
- Sim, API, CLI, and test fixture consumers all migrated.

**Tier 2 — Cross-cutting primitives (5 substeps):**
- **Pill:** shared filter toggle button.
- **WhyLink:** explainability affordance (labeled and dot variants),
  navigates to `/explain` with subject params.
- **TeamMark:** refactor with team primary color background via
  React Query cache; falls back to grey when unavailable.
- **Spark:** generic sparkline generalized from `RatingHistorySparkline`.
- **TeamHero:** composed team identity block (team-colored mark + city +
  name + record + rating), left/right orientation.

**Tier 3 — Dashboard sections (5 substeps):**
- **FeaturedMatchupsGrid:** 3-card top row. Composes `/edges` (ranked
  by EV) + `/games` (predictions). Uses TeamHero, WinProbBand, WhyLink,
  bet slip integration.
- **ModelEdgesTable:** sortable ranked table with 4 filter tabs
  (All/Spread/Total/Moneyline). Uses Pill primitive.
- **PropEdgesRail:** 5-row compact list sorted by predicted mean
  descending. WhyLink and slip integration.
- **ModelPerformanceRail:** Spark-based sparkline + all-time ROI +
  W-L-P record + bankroll CTA. Consumes `/portfolio/summary` and
  `/portfolio/curve`.
- **Dashboard integration:** 2-row grid layout. Removed API loop
  verification card and field-status demo (moved to git history only).

### Architecture established

- **`components/primitives/` folder** for cross-domain shared components.
  Future workstreams pull from here.
- **`components/dashboard/` folder** for Dashboard-specific sections.
  Pattern reusable for other screen rebuilds.
- **Team color hook (`useTeamMetadata`, `useTeamByAbbr`):** React Query
  cache with 5-minute stale time. All screens using `TeamMark` benefit.
- **Placeholder odds (-110):** when adding edges to bet slip from Dashboard,
  we use -110 as placeholder odds until W7 multi-book lands.

### Test coverage

59 tests total across primitives + Dashboard sections. Vitest + React
Testing Library. Every primitive has its own unit test file.

### What's not shipped (surfaced during work)

- Rolling 7d/30d ROI windows on `/portfolio/summary` — tracked in
  ROADMAP §9.7. Currently shown as "All-Time" honestly.
- Real market odds — bet slip integration uses placeholder -110. Blocked
  on W7 (multi-book odds).
- Prop leg semantic — `add()` currently encodes prop as "prop" market
  type; BetLeg schema might need refinement for prop legs specifically.

### Next

Between workstreams. Options: W12 (Model Ensemble), W7 (Multi-Book Odds),
W4.5 (Scenario Engine, blocked on §5.3), or another frontend polish
sweep pulling from §9.7/§9.8 backlogs.

## 2026-07-04 — W8 Tier 3: Additive Datasets (7 additives, 15+ substeps)

Seven-step tier closing out W8 Tier 3. Each additive is a small feature
engineering module + persistence artifact + loader + serializer wiring
that populates one or more previously-scaffolded `field_status: pending`
fields on the API.

### Additives shipped

- **Step 1 (2026-07-03):** `week_over_week_delta` on `/projections`. No new module — reads directly from Elo state table.
- **Step 2 (2026-07-03):** Per-team percentile ranking pass (4 stats: elo, avg_wins, make_playoffs, win_sb). Populates `/teams`, `/teams/{abbr}`, `/compare/teams`. New module `evaluation/percentiles.py`.
- **Step 3 (2026-07-03):** `trend` on `/teams` and `/teams/{abbr}`. Reused `compute_elo_deltas` from Step 1.
- **Step 4 (2026-07-03):** `n_simulations` on `/projections`. New `projections_metadata.json` sidecar written by `run_full_simulation`.
- **Step 5 (2026-07-03):** Per-player situational splits (8 cohorts) on `/props/{prop_id}`. Joins player game logs to games CSV. New module `evaluation/situational_splits.py`.
- **Step 6 (2026-07-04):** Opponent-allowed-by-position defense rows on `/compare/player/{prop_id}` (3 of 4 rows; `red_zone_rate_allowed` deferred). New module `evaluation/opponent_allowed.py`.
- **Step 7 (2026-07-04):** Team cohort splits (4 cohorts × 8 metrics) on `/compare/teams`, `/teams/{abbr}`, `/games/{game_id}`. New module `evaluation/team_cohort_splits.py` and new `gridiron teams` CLI subcommand.

### New CLI subcommands

- `gridiron sim compute-percentiles`
- `gridiron props compute-splits`
- `gridiron props compute-opponent-allowed`
- `gridiron teams compute-cohort-splits`

### New persistence artifacts

- `data/output/rankings/percentiles/percentiles_{season}_wk{NN}.parquet`
- `data/output/rankings/team_cohort_splits.parquet`
- `data/output/props/situational_splits/{stat_type}.parquet`
- `data/output/props/opponent_allowed.parquet`
- `data/output/temp/projections_metadata.json`

### Test-fixture inconsistencies discovered

- `MiniRepoBuilder.with_teams_reference()` produces modern short codes (`KC`, `LAC`, `BUF`, `MIA`) but rest of codebase uses PFR-era codes (`KAN`, `JAC`). Not blocking. Captured in ROADMAP §9.6.

### Remaining not-shipped

- Off/def rating decomposition (real modeling work).
- Various `field_status: pending` fields blocked on named workstreams.

### Next

W8 workstream closed. Between-workstreams pause. Available next workstreams: W12 (Ensemble), W4.5 (Scenario, blocked on §5.3), W7 (Multi-Book, blocked on §5.2), W10 (Real-Time, deferred).

## 2026-07-03 — W9 Frontend (20-screen React app consuming the API)

Three-tier workstream shipping a complete React frontend. Consumes
the 16-endpoint W8 Tier 2 API surface end-to-end. Every prototype
screen renders with real data where populated, structured
`field_status` where scaffolded, and consistent error UX everywhere.

### Shipped

**Tier 1 — Client infrastructure (7 substeps):**
- Vite + React + TypeScript scaffolding with Geist font loading and
  OKLCH dark theme port.
- Chrome components: TopNav, SubNav, Breadcrumb.
- Three React Contexts (AppState, BetSlip, Nav) with
  localStorage/sessionStorage persistence.
- `openapi-fetch` typed API client from checked-in schema.
- React Query with per-endpoint hooks and query key namespacing.

**Tier 2 — Populated screens (7 substeps + 1 pre-substep):**
- Pre-substep 2.0: Field-status primitives (`<PendingField />`,
  `<BlockedField />`, `<FieldValue />`).
- 2a: Games (GamesList + GameDetail).
- 2b: Teams (TeamRankings + TeamProfile).
- 2c: Projections (PlayoffProjections).
- 2d: Players/Props (PlayersExplorer + PlayerProp).
- 2e: Compare (ComparePage) with URL-synced state for bookmarking.
- 2f: Bankroll consuming 5 `/portfolio/*` endpoints in parallel.
- 2g: BetSlip staging bets from `/edges`.

**Tier 3 — Blocked screens + polish (6 substeps):**
- 3a: BlockedScreen for LineShopping, LiveGame, NewsWire, ExplainPage.
- 3b: Settings, Onboarding, Tools (client-side, no API).
- 3c: Aesthetic identity documented.
- 3d: Vitest smoke tests.
- 3e: Keyboard accessibility sweep.
- 3f: ErrorCard + OfflineBanner.

### Architecture established

- **Data flow:** endpoint → generated TypeScript type → React Query hook
  → screen component → shared primitive (FieldValue, ErrorCard, etc.).
- **State:** three Contexts (Nav, BetSlip, AppState) persisted client-side.
- **Styling:** OKLCH dark theme via CSS variables. Cards for grouping,
  monospace for numerics, serif for editorial emphasis.
- **Testing:** Vitest + React Testing Library. Smoke coverage of
  critical paths.
- **Accessibility:** Semantic HTML, focus indicators, ARIA labels on
  icon-only controls.

### W8 backend hygiene items surfaced

- `season` type inconsistency (int vs string across endpoints).
  Captured in ROADMAP §9.6.
- Team abbreviation convention (KAN/JAC vs KC/JAX). Not blocking, not
  tracked as an issue.
- `evaluate select-model --write-manifest` display bug: "Persist
  manifest" step appeared after the summary block. Fixed as a small
  W8 patch during 2f verification.

### Next

W8 Tier 3 designing. Now that W9 has surfaced which pending/blocked
states appear on real screens, the additive dataset priority can be
decided empirically rather than speculatively. Most likely first
additive: per-stat percentile ranking (drives compare screen rank
columns and team detail rank fields) OR opponent-allowed-by-position
(drives the entire defense side of PlayerProp).

## 2026-07-02 — W8 Tier 2: Direct-Serialization Endpoints (16 endpoints populated)

Eight-step tier closing out Tier 2 of the API serving layer. Every
prototype-referenced URL returns a 200 with a Pydantic-validated
shape. Fields not yet populated are marked with structured
`_meta.field_status` per D14.

### Endpoints populated

Step 1 (2026-07-01): `/weeks/current` and all `/portfolio/*`.
Step 2 (2026-07-01): `/model/performance` (composed metrics endpoint).
Step 3 (2026-07-01): `/teams` and `/teams/{abbr}`.
Step 4 (2026-07-01): `/projections`.
Step 5 (2026-07-01): `/games` and `/games/{game_id}`.
Step 6 (2026-07-01): `/edges`.
Step 7 (2026-07-01): `/props` and `/props/{prop_id}`.
Step 8 (2026-07-01): `/compare/teams` and `/compare/player/{prop_id}`.

### Architecture established

- **Loader pattern (`api/loaders.py`):** pandas DataFrames in, dicts
  out, explicit `settings.repo_root` threading (D19).
- **Schema pattern (`api/schemas/*.py`):** Pydantic v2, `frozen=True`,
  `extra="forbid"`, nullable defensive fields, `_meta` envelope
  via `BaseResponse` / `BaseListResponse`.
- **Serializer pattern (`api/serializers/*.py`):** hand-written per
  D17, owns `_meta.field_status` construction per D18.
- **Route pattern (`api/routes/*.py`):** FastAPI, exception
  translation (`ChampionNotFoundError` → `NO_CHAMPION_MANIFEST`,
  `OddsUnavailableError` → `NO_ODDS_AVAILABLE`), lazy scope resolution
  (Step 7d learning).
- **Testing pattern:** `MiniRepoBuilder` extended with four
  W8-specific methods (`with_champion_manifest`,
  `with_predictions_archive`, `with_odds_snapshot`,
  `with_teams_reference`); integration tests via FastAPI
  `dependency_overrides`.

### New `Unavailable` slugs registered

`NO_CHAMPION_MANIFEST` (Step 5d), `NO_ODDS_AVAILABLE` (Step 6d),
`OPPONENT_ALLOWED_BY_POSITION` (Step 8b).

### New `api/` modules

- `api/exceptions.py` — API-surface data-state exceptions from
  loaders to routes.
- `api/_prop_id.py` — Shared `decode_prop_id` helper used by
  `/props/{prop_id}` and `/compare/player/{prop_id}`.

### Field_status scaffolding

Fields not yet populated ship with structured `_meta.field_status`
metadata. Categories:

- **Pending backend work:** kick, venue, weather (games); line, p_over,
  lean, confidence_tier (props); schedule_difficulty,
  playoff_probability, cohort_splits, percentile_ranks (teams
  compare); recent_form, situational_splits, historical_vs_opponent
  (props/compare).
- **Blocked on named workstreams:** swing_factors, prop_reasoning
  (feature attribution); injuries, injury_status (§5.3);
  multi_book_shopping (W7); off_rating, def_rating (Tier 3);
  trend (weekly Elo snapshot); avg_allowed, rank_against_position,
  last_5_games_avg, red_zone_rate_allowed (opponent aggregation).

### Tests

- Per-route integration test file in `tests/integration/api/` for
  each populated endpoint cluster (games, edges, props, compare).
- Per-schema unit test file in `tests/unit/api/` for each new schema.
- Per-serializer unit test file for each new serializer.

### Next

Tier 3 (additive datasets) designing. Kickoff waits for W9
(Frontend) feedback to identify which additive to build first.
W9 unblocked and ready to start.

## 2026-07-01 — W13 Tier 3: CLI Consumer Refactor (W13 workstream complete)

Four-step tier migrating CLI consumers to use the champion manifest
via the ``--model-type auto`` sentinel pattern. Closes W13 as a
workstream.

### Shipped
- ``cli/_composites.py::resolve_win_prob_model_type`` — helper for
  the ``"auto"`` sentinel. Reads the manifest; passes explicit values
  through; raises ``typer.BadParameter`` on missing manifest with an
  actionable message.
- ``cli/weekly_predict.py`` — ``--model-type`` default flipped from
  ``"random_forest"`` to ``"auto"``. Resolution happens after
  Typer/user-input validation.
- ``cli/edges.py`` — both ``report`` and ``clv`` migrated with the
  same pattern.
- Intentional Elo callsites annotated with comments explaining why
  they aren't migrated:
  * ``cli/weekly_predict.py::_stage_predict_week`` (archives
    ``build_predictions_df`` output, which is Elo-based).
  * ``cli/output.py::output_predictions`` (same pattern).
  * ``cli/evaluate.py::evaluate_tune`` — both ``--apply`` branches
    (tune is Elo-specific by design).
  * ``cli/evaluate.py::evaluate_backfill`` — CLI defaults are
    historical convenience, not a champion pick.
  * ``cli/ratings.py::elo_evaluate`` — Elo command by name.

### Tests
- ``tests/unit/cli/test_composite.py`` — added
  ``TestResolveWinProbModelType`` (4 tests).
- ``tests/unit/cli/test_weekly_predict.py`` — added
  ``TestModelTypeResolution`` (3 tests). Existing
  ``test_runs_all_stages_when_all_succeed`` updated to pass
  ``--model-type random_forest`` explicitly.
- ``tests/unit/cli/test_edges.py`` — new file with
  ``TestReportModelTypeResolution`` and ``TestClvModelTypeResolution``
  (3 tests each).
- ``tests/integration/test_edges_cli.py`` — six existing tests
  updated to pass ``--model-type random_forest`` explicitly.

### Scope note

The original W13 handoff paragraph identified "8 hard-coded
callsites." Categorization during Tier 3 design revealed that only
3 were user-facing CLI defaults that should resolve to the champion
(``weekly_predict``, ``edges report``, ``edges clv``). The other 5
were:

- Provenance labels for Elo-based predictions (correct as-is).
- Elo-specific by design (correct as-is).
- Historical CLI convenience defaults (kept for backward compat;
  users pass explicit values in practice).

All 5 got explanatory comments instead of refactors.

### W13 workstream summary (Tiers 1–3, all shipped 2026-07-01)

- **Tier 1:** manifest schema + reader API (``champion_resolver.py``).
- **Tier 2:** writer + full-retrain integration + manual-override
  CLI flags (9 steps).
- **Tier 3:** CLI consumer migration (4 steps).

### Next
W8 (API Serving Layer) resumes at Tier 2 Step 5. Runtime champion
resolution now available for the ``/games``, ``/games/{id}``,
``/games/{id}/predictions``, ``/edges``, and ``/props/{prop_id}``
endpoints via ``resolve_current_champion``.

## 2026-07-01 — W13 Tier 2: Runtime Champion Manifest (Writer + Full-Retrain Integration)

Nine-step tier closing out the writer half of Runtime Champion Resolution.
The static manifest at ``data/output/champions/champions.json`` is now
populated by the ``promote-champions`` stage in ``full-retrain`` and by
optional ``--write-manifest`` flags on ``evaluate select-model`` and
``props champion``. All champion decisions across CLI and stage
surfaces share the same code path.

### Shipped
- ``champion_resolver.write_manifest`` — atomic write via
  ``os.replace``; preservation semantics for per-entry ``source_run_id``.
- ``evaluation.champion.select_game_classification_champions`` —
  wraps ``select.py``'s ``collect_model_metrics`` + ``rank_models``
  on Brier / ECE / AUC.
- ``evaluation.champion.select_game_regression_champions`` — reads
  ``ArtifactStore`` metadata; picks lowest MAE; tie-breaks to
  ``random_forest``.
- ``evaluation.champion.select_prop_champion_for_family`` and
  ``select_prop_champions_all_families`` — iterate ``PropModelType``,
  build ``RegressionModelResult`` per algorithm from the prop archive,
  delegate to existing ``select_prop_champion``.
- ``evaluation.champion.build_prop_champion_candidates`` — shared
  per-algorithm evaluation helper reused by the selector and by
  ``props champion`` for terminal display.
- ``evaluation.champion.promote_champions`` — pure function combining
  the three selectors + manifest merge + write. Returns a
  ``PromoteChampionsResult`` with fresh, preserved, and warning fields.
- ``cli/full_retrain.py::_stage_promote_champions`` — thin adapter over
  ``promote_champions``. Depends on ``refresh-calibrations`` only;
  runtime order still places it after ``backfill-prop-models`` when
  both are active. ``baseline-report`` re-wired to depend on
  ``promote-champions``.
- ``cli/full_retrain.py::_stage_baseline_report`` — appends a Current
  Champions bullet-list block above the Game Models table. Format
  chosen so the existing markdown-table delta parser ignores it.
- ``cli/_composites.py::write_champion_manifest`` — shared helper for
  the manual-override CLI flags.
- ``cli/evaluate.py::evaluate_select_model`` — new ``--write-manifest``
  flag. Runs the full catalog through ``promote_champions``.
- ``cli/props.py::champion_cmd`` — new ``--write-manifest`` flag.
  Refactored inline per-algorithm loop to use
  ``build_prop_champion_candidates``.
- ``models/catalog.py`` — new module. Single source of truth for
  ``GAME_MODEL_PAIRS``, ``PROP_STAT_FAMILIES``, ``PROP_ALGORITHMS``.
  Used by both ``full_retrain.py`` and the manual-override flags.

### Decisions

No new architectural decisions locked at the D-level; all decisions
were within the Tier 2 design phase and are captured in PLAN.md's
inline "How" block for the tier.

### Tests

- ``tests/unit/evaluation/test_champion_resolver.py`` — 8 tests for
  ``write_manifest`` (schema, atomic write, preservation semantics,
  roundtrip, empty writes, defensive copy).
- ``tests/unit/evaluation/test_champion.py`` — added
  ``TestSelectGameRegressionChampions`` (9 tests),
  ``TestSelectGameClassificationChampions`` (9 tests),
  ``TestSelectPropChampionForFamily`` (5 tests),
  ``TestSelectPropChampionsAllFamilies`` (3 tests),
  ``TestPromoteChampions`` (3 tests),
  ``TestBuildPropChampionCandidates`` (3 tests).
- ``tests/unit/cli/test_full_retrain.py`` — added
  ``TestStagePromoteChampions`` (5 tests), extended ``TestStageList``
  (2 new tests), extended ``TestBaselineReportStage`` (3 new tests),
  updated ``TestCommandInvocation`` for the new stage.
- ``tests/unit/cli/test_evaluate.py`` — new file with
  ``TestSelectModelWriteManifestFlag`` (3 tests).
- ``tests/unit/cli/test_props_champion_write_manifest.py`` — new file
  with ``TestPropsChampionWriteManifestFlag`` (3 tests).

### Next
Tier 3: refactor the 8 hard-coded ``model_name="win_prob",
model_type="elo"`` callsites across ``weekly_predict.py``, ``output.py``,
``edges.py``, ``evaluate.py`` to use ``resolve_current_champion``.
Confirm ``ratings.py``'s intentional Elo usage stays as-is with a
comment.

## 2026-06-27 — W8 Tier 1: API Skeleton and Blocked-Endpoint Stubs

First tier of W8. FastAPI app skeleton at `src/gridiron_edge/api/` plus
12 blocked-endpoint stubs matching the prototype URL inventory. Each
returns 200 with a structurally valid null response carrying registered
blocker slugs. Reachable via `gridiron api serve`.

### Shipped

- `api/app.py` — FastAPI factory with OpenAPI tag inventory and CORS.
- `api/meta.py` — `ResponseMeta`, `Blocker` registry, `Unavailable` slugs.
- `api/schemas/_base.py` — `BaseResponse` / `BaseListResponse` with
  `_meta` envelope.
- `api/deps.py` — `SettingsDep` / `DataPathResolverDep` shared
  dependencies with a single override seam.
- Twelve stubbed routes for Tier 3-blocked endpoints (comparables,
  explain, injuries, lines, live, model, news, prop_reasoning,
  prop_shop, swing_factors, plus placeholder shapes for teams,
  projections). Each returns a structurally valid null response
  with `_meta.field_status` populated.

### Architectural decisions

- **D14:** Placeholder convention — null field + `_meta.field_status`
  entry with either the literal string "pending" or a `BlockedStatus`
  object naming a stable blocker slug.
- **D16:** Every Tier 3 route uses a slug registered in
  `Blocker.all_slugs()`; consistency test enforces this.

### Tests

- Integration tests confirm all 12 endpoints reachable via
  `gridiron api serve`, return the expected status codes, and carry
  valid `_meta.field_status` metadata where scaffolded.

### Next

Tier 2 (populated endpoints): fill in the 16 endpoints the prototype
consumes with real data. Establish loader/schema/serializer/route
pattern.

## 2026-06-22 — Workstream 5: Tier 4 Cleanup Sweep

### Summary

Multi-session opportunistic cleanup that closed 30 items from the Tier 4 backlog and surfaced two real bugs that were promoted to fixes during the sweep. The Tier 4 backlog is retired; remaining items moved to PLAN.md as workstream candidates.

### Highlights

**CLI ergonomics (4 items):**
- `bet summary` now renders `calibration_health` and `ev_vs_actual_gap` from the existing summary dict
- `models info win_prob elo` now directs analytic-model users to `evaluate summary` instead of suggesting training
- `props train-and-save` exposed as a CLI command to produce persisted artifacts for projections
- CLI season-label inconsistency resolved — `props backfill` now accepts both `2023` and `"2023-2024"` formats

**Composite commands (5 items):**
- `weekly-predict` renders top-edge preview from ranked edge report
- `full-retrain` generates timestamped baseline reports with delta-vs-previous tables
- `verify` baseline-comparison now actually compares metrics against the latest full-retrain
- `verify` composite-key parser correctly handles multi-token model types (e.g., `random_forest`)
- `full-retrain` calibration values persist to disk at `data/output/calibration/game_model_calibration.json`
- `post-week` drift threshold extracted to a named constant

**Dead code removal (5 items):**
- `_shared.py` re-export shim
- `_game_location` helper (logic was inlined into the cleaner code path)
- `_EPA_RELIABLE_FROM` constant
- `UNIVERSAL_FEATURE_COLS` and the related test fixtures
- `PropPrediction` dataclass (vestigial pre-DataFrame design)
- `max_mae_tolerance` field in `RegressionPromotionGates` (defined but never enforced)

**Documentation drift (3 items):**
- Stale `WS2` / `D1` / `D3` workstream markers removed from production source
- Schema version comments referencing v2/v3 replaced with version-neutral language
- Phase markers like `(existing)` / `(new)` audit complete; surviving instances refer to runtime state, not project history
- HTML escaping added to `viz/predictions.py::render_predictions_html`

**Architecture (4 items):**
- `_TEAM_CODE_MAP` historical abbreviation mapping consolidated into `core/constants.py::TEAM_CODE_NORMALIZATION`
- `run-data-pipeline` retained as a data-layer primitive (intentionally not refactored to composite form)
- `repos.py::with_epa_by_game` routed through the shared `_write()` helper for registry consistency
- Inline imports in composite CLI files: lightweight imports moved to module top, heavy imports (matplotlib, sklearn-touching, prediction pipeline, prop registry) kept inline for fast `gridiron --help` startup

**Type safety and error handling (2 items):**
- Exception narrowing in viz/predictions.py (GAMETIME parsing), ingest/odds/draftkings.py (float coercion), cli/betting.py (odds ledger load). Broad catches retained where defensive
- `# pyrefly: ignore` and `Any` annotation audit complete. Most existing suppressions are legitimate workarounds for known stub limitations; further type work deferred

### Real bugs surfaced and fixed

**XGBoost recalibration Pipeline feature-name warning:** The `CalibratedClassifierCV` Pipeline in the XGBoost post-training calibration branch was fitted on a DataFrame and predicted on `.values` arrays, producing sklearn `UserWarning` at every predict call. Fix: fit and predict on `.values` arrays consistently with the rest of the codebase.

**Modeling file stale-data preservation:** Investigation of a row-count discrepancy between feature sets revealed that the incremental build mode of `build_model_inputs()` was silently preserving stale weather data for ~12,000 historical rows. Weather data was missing for 1999-2010 seasons in the modeling file despite the weather source data being complete and the WeatherFeature implementation being correct. Root cause: incremental builds only recomputed features for new GAME_IDs, leaving older rows with values from whenever they were first computed. Fix: added `data_version` field to the modeling manifest; pipeline detects mismatch and forces a full rebuild with a warning. Convention documented for future bug fixes.

### What this enables

- Tree-based game models can now train on the rebuilt modeling file with 9,920 training rows (up from 5,705, a 74% increase), thanks to historical weather data now being available
- Future feature implementation bug fixes will trigger automatic full rebuilds via `data_version` bumps, preventing silent stale data
- The composite CLI workflows produce richer terminal output (top edges in weekly-predict, drift health in post-week, baseline diffs in full-retrain, real metric comparison in verify)
- Disk-backed calibration values persist across `full-retrain` runs

### What's deferred to future workstreams

- **Testing infrastructure** (5 items): props e2e tests, composite commands e2e tests, weather ingest integration test, registry cold-start scenarios, performance baselines
- **Real bug** (1 item): Walk-forward backfill produces no valid pipeline for single-season windows with expanded feature sets
- **Investigation** (1 item): `CalibratedClassifierCV` shuffle=False → TimeSeriesSplit comparison
- **Operational** (4 items): DraftKings 403, stadium coverage data entry, calibration refresh after next full-retrain, `verify --strict` CI gate

### Files retired

- `TIER_4_BACKLOG.md` — replaced by PLAN.md's "Future Workstream Candidates" section

## 2026-06-18 — Workstream 2: Game Model Refactor

### Added
- `BaseModelMetadata` shared metadata type with `GameModelMetadata` and `PropModelMetadata` subclasses.
- `GamesTrainer` + spec subclasses (`WinProbTrainer`, `TotalTrainer`) for unified game model training.
- `GamesPredictor` base class with five composite-key registrations.
- Composite registry keys (e.g. `win_prob_random_forest`) replacing flat keys.
- Nested artifact path scheme `data/models/{model_name}/{model_type}/`.
- Elo migrated to `win_prob_elo` composite registration.

### Changed
- All classification metrics (Brier, ECE, AUC, log_loss, accuracy) are now first-class fields on `GameModelMetadata`.
- Prediction archive schema migrated from `model_version` to `(model_name, model_type)`.
- CLI commands use `(model_name, model_type)` pair throughout.

### Removed
- `models/game_prediction/tree.py`, `logistic.py`, `pipeline.py`.
- Free functions `train_total_model`, `load_total_model`, `predict_total`.
- Flat registry keys: `logistic`, `random_forest`, `xgboost`.
- Legacy `LogisticPredictor`, `RandomForestPredictor`, `XGBoostPredictor` re-exports.
- `EloV1Predictor`, `EloV2Predictor`, `EloV3Predictor` (collapsed into `WinProbEloPredictor`).
- `evaluation/archive.py::migrate_archive` (no longer needed).

## 2026-06-17 — Workstream 1: Champion/Challenger for Props

- **Prop model factory pattern** (`PropModelType` enum: elasticnet,
  random_forest, xgboost) with `_create_model()` factory and
  `_get_param_grid()` providing per-algorithm HP search spaces
  (ElasticNet: 25 combos, RandomForest: 36, XGBoost: 54).
- **Spec-only subclasses**: all 5 prop trainers (`qb_pass_yards`,
  `qb_rush_yards`, `rb_rush_yards`, `wr_rec_yards`, `te_rec_yards`)
  reduced to ~15-20 lines each. Shared `_fit()`, `_predict()`,
  and `train(model_type=)` consolidated in `PropTrainer` base.
- **`clip_lo` / `clip_hi` on `PropModelSpec`**: spec-driven prediction
  clipping (0.0 floor; per-position ceilings of 200-600 yards).
- **`model_type` field on `PropModelMetadata`**: artifact tracking.
- **Generalized champion/challenger gates** in `evaluation/champion.py`:
  - Classification path (game models) renamed for symmetry:
    `ClassificationPromotionGates`, `ClassificationComparisonResult`,
    `extract_classification_metrics`, `compare_classification_models`,
    `format_classification_comparison`.
  - Regression path (prop models): `RegressionPromotionGates` (R² > 0,
    coverage in [0.85, 0.97]), `RegressionModelResult`,
    `RegressionComparisonResult`, `compare_regression_models()`,
    `select_prop_champion()` (lowest MAE among eligible, ElasticNet
    fallback per Decision #11), `format_regression_comparison()`.
- **CLI enhancements** in `cli/props.py`:
  - `evaluate --model-type {elasticnet,random_forest,xgboost}`
  - New `champion` command - trains all 3 types, compares, selects
  - `console.header()` / `step()` / `console.summary()` parity with
    game model CLI; tqdm bars match game model styling
    (ncols=88, colour="cyan", live best-metric postfix).
- **Tests**: 15 prop champion tests (factory, grids, clips), 16
  regression champion gate tests, 5 CLI structure tests, plus
  updates to existing classification tests for renamed symbols.

### Validated

- All 15 prop model trainings completed via
  `gridiron props champion --model all`.
- ElasticNet selected as champion in 5/5 stat families.
- `qb_rush_yards` triggered fallback policy (no model passes R²>0
  guardrail) - known limitation; feature work deferred to later WS.

### Changed

- `cli/models.py`: updated to use renamed classification symbols
  (`ClassificationComparisonResult`, `compare_classification_models`,
  `extract_classification_metrics`).

## 2026-06-10 - W4: Player Data & First Prop Models - Mostly Complete

Complete player-level data pipeline, 5 trained prop models, post-processing
enrichment, evaluation metrics, archive, and CLI. M3 milestone achieved.

##### Player data foundation (Phase A)
- **nflreadpy migration:** Switched from archived nfl_data_py to nflreadpy.
  Key API changes: import_weekly_data() → load_player_stats().to_pandas().
  nflreadpy returns Polars DataFrames requiring .to_pandas() conversion.
- **Player stats ingest:** 26 seasons (1999–2024), ~5K rows/season,
  42 columns per player-game row. Stored at data/raw/player_stats/.
- **Player stats cleaning:** Dropped rows with null game_id (1 row,
  1999 week 9), deduplicated 46 schedule-join mismatch rows.

##### Player feature engineering (Phase B)
- **Rolling features (features/player/rolling.py):** L3 and L6 rolling
  mean + std for 23 stat columns (~46 features). Shift(1) prevents
  lookahead. Position-specific stat columns.
- **Matchup features (features/player/matchup.py):** 28 features -
  14 defensive-allowed stats × 2 (L6 rolling average + rank).
  Rankings: 1=toughest, 32=most generous. Joined via opponent_team.
- **Usage features (features/player/usage.py):** 6 features -
  target_share, carry_share, touch_share × L3/L6 windows.
- **Game context features (features/player/game_context.py):** 6 features
  from cleaned games CSV - is_home, game_spread, over_under,
  implied_team_total, is_dome, rest_days. No shift(1) needed -
  these are known pre-game.
- **Unified builder (features/player/builder.py):**
  build_prop_features(position_filter=["QB"]) chains all 4 builders
  with single parquet load. NaN handling deferred to trainer.
- **Programmatic feature list (features/player/_columns.py):**
  PROP_FEATURE_COLS built from component modules - stays in sync.

##### Prop model training (Phase C)
- **PropTrainer base class (models/prop_prediction/base.py):**
  - _load_data() calls build_prop_features()
  - train() uses HOLDOUT_SEASONS (2023–2025) - consistent with game models
  - Position-aware NaN handling: features with >50% NaN for the filtered
    position are dropped before training
  - _feature_columns() returns PROP_FEATURE_COLS
  - _build_features() non-abstract - default returns df as-is
  - Deleted dead _join_game_context() and _join_schedule_context()
- **5 trained models (ElasticNet baselines):**

| Model | Train | Holdout | MAE | RMSE | R² | Nonzero |
|-------|-------|---------|-----|------|----|---------|
| qb_pass_yards | 5,706 | 1,367 | 58.0 | 72.6 | 0.071 | 37/128 |
| qb_rush_yards | 1,434 | 468 | 16.4 | 20.2 | 0.090 | 52/128 |
| rb_rush_yards | 10,023 | 2,001 | 25.0 | 32.3 | 0.168 | 16/124 |
| wr_rec_yards | 23,831 | 4,535 | 25.1 | 32.9 | 0.203 | 55/120 |
| te_rec_yards | 10,087 | 2,052 | 18.3 | 24.2 | 0.188 | 58/120 |

##### Post-processing enrichment (Phase C2)
- **models/prop_prediction/post_process.py:** Pure function architecture.
  - predicted_std = sqrt(model_rmse² + player_L3_std²)
  - 90% prediction intervals, lo_90 clipped at 0
  - P(over) = 1 - Φ((line - mean) / std), Normal CDF for V1
  - Lean: Over (>0.55), Under (<0.45), No Edge
  - Confidence tier: High (|p-0.5|>0.15), Moderate (>0.08), Low
  - Line input optional - NaN line → NaN p_over/lean/tier
  - TARGET_STD_MAP maps model names to rolling std columns

##### Evaluation metrics (Phase D1)
- **evaluation/prop_metrics.py:** 6 metric functions + orchestrator.
  - AccuracyMetrics: MAE, RMSE, R², median AE
  - BiasMetrics: mean error, % over-predicted
  - CoverageMetrics: actual vs nominal coverage, interval width
  - CalibrationMetrics: P(over) reliability diagram, MACE
  - HitRateMetrics: Over/Under/overall, push exclusion
  - TierMetrics: per-tier MAE, hit rate, |p_over - 0.5|
  - PropEvalReport.print_summary() formatted output
  - Graceful degradation: accuracy/bias always; others when data available

##### Prop archive (Phase D2)
- **evaluation/prop_archive.py:** Append-only Parquet.
  - 19-column schema, dedup on (game_id, player_id, stat_type, model_version)
  - Metadata: predicted_at, is_backfilled, model_version
  - Optional filters on load: stat_type, season

##### Prop CLI (Phase E1)
- **cli/props.py:** 3 commands registered as gridiron props.
  - gridiron props evaluate --model qb_pass_yards
  - gridiron props projections [--model all] [--top 20]
  - gridiron props backfill --model qb_pass_yards
  - Lazy trainer registry for fast --help
  - _train_and_enrich() shared helper

##### First CLI evaluation (qb_pass_yards)
- MAE: 63.4, RMSE: 80.6, R²: 0.118, Median AE: 51.8, N: 1,433
- Bias: +9.7 (over-predicting), 52.8% over-predicted
- Coverage: 93.8% (nominal 90%), interval width: 323.6

##### Key design decisions
- Position-aware NaN handling (>50% threshold per position)
- predicted_std combines model RMSE + player L3 rolling std
- No shift(1) on game context features (known pre-game)
- Builder does NOT dropna - trainer handles with position context
- PROP_FEATURE_COLS built programmatically from component modules
- Spread derived from FAVORITED + abs(VEGAS_LINE)
- Normal CDF for P(over) V1 - upgradeable to empirical later
- Lean/tier thresholds consistent with game model post-processing

##### Deferred
- E2: DraftKings prop odds ingest
- Champion/challenger for props (RF, XGBoost)
- Integration/E2E tests for prop pipeline
- Snap % features (nflreadpy doesn't expose snap counts)

##### Tests added: ~90 new
- test_prop_post_process.py (26 tests, 7 classes)
- test_prop_archive.py (16 tests, 3 classes)
- test_prop_metrics.py (23 tests, 7 classes)
- test_builder.py (unit tests for unified builder)
- test_qb_rush_yards.py (5 tests)
- Updates to test_qb_pass_yards, test_rb_rush_yards, test_wr_rec_yards,
  test_te_rec_yards (PROP_FEATURE_COLS migration)

##### Files added

| File |
|------|
| src/gridiron_edge/features/player/builder.py |
| src/gridiron_edge/features/player/_columns.py |
| src/gridiron_edge/features/player/game_context.py |
| src/gridiron_edge/features/player/usage.py |
| src/gridiron_edge/models/prop_prediction/post_process.py |
| src/gridiron_edge/models/prop_prediction/qb_rush_yards.py |
| src/gridiron_edge/evaluation/prop_metrics.py |
| src/gridiron_edge/evaluation/prop_archive.py |
| src/gridiron_edge/cli/props.py |
| tests/unit/features/test_builder.py |
| tests/unit/models/test_prop_post_process.py |
| tests/unit/models/test_qb_rush_yards.py |
| tests/unit/evaluation/test_prop_metrics.py |
| tests/unit/evaluation/test_prop_archive.py |

##### Files modified

| File | Change |
|------|--------|
| src/gridiron_edge/models/prop_prediction/base.py | Rewired to build_prop_features, HOLDOUT_SEASONS, position-aware NaN |
| src/gridiron_edge/models/prop_prediction/qb_pass_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/rb_rush_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/wr_rec_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/te_rec_yards.py | Removed _build_features override |
| src/gridiron_edge/features/player/rolling.py | Added optional df param |
| src/gridiron_edge/features/player/matchup.py | Added optional df param, fixed line length |
| src/gridiron_edge/cli/main.py | Registered props_app |

##### Summary
- **9 new source files**, 8 modified
- **5 new test files**, 4 modified, **~90 new tests**
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-04 - Sigma/Margin_std Recalibration & Versioned Model Cleanup - Complete

Recalibrated spread derivation parameters and confidence tiers after
TimeSeriesSplit retrain. Cleaned all vestiges of old versioned model names.

##### Sigma/margin_std recalibration
- Calibrated on holdout seasons (2023–2025) using existing
  calibrate_spread_sigma() infrastructure
- random_forest: sigma 13.97→10.63, margin_std 12.85→13.54
- xgboost: sigma 13.95→11.43, margin_std 13.44→13.34
- logistic: sigma 12.75→11.99, margin_std 13.53→13.29
- Spread ranges compressed (e.g. RF [-43, 16] → [-33, 12])
- Old sigmas were inflated because StratifiedKFold CV pushed
  models toward overconfident probabilities

##### Confidence tier rework
- Old approach: band_width (win_prob_hi - win_prob_lo) thresholds
  at 0.65/0.82. With honest margin_std, band_width was nearly
  constant (~0.95 for all games), making tiers useless (98.7% Low)
- New approach: probability distance from 0.5, folded to favorite
  side. Thresholds: >= 0.70 High, >= 0.60 Moderate, else Low
- Uses max(prob, 1-prob) to avoid IEEE 754 subtraction artifacts
- Validated win rates: High ~80%, Moderate ~65%, Low ~54%
- Distribution: ~23% High / ~30% Moderate / ~47% Low

##### Versioned model cleanup
- Removed all versioned entries (rf_v1–v3, xgb_v1–v3, logistic_v1–v4,
  elo_v1) from _MODEL_SIGMAS and _MODEL_MARGIN_STDS dicts
- Cleaned prediction archive of ~90K old versioned-model rows
- Updated all versioned model references in tests, docstrings,
  comments, and diagnostics colors to unversioned champion names
- total_rf_v1 references intentionally preserved (not part of
  champion/challenger system)

##### Scripts added
- scripts/recalibrate_sigma.py - holdout sigma/margin_std calibration
- scripts/clean_archive.py - archive cleanup for deprecated model versions

##### Files changed

| Action | File |
|---|---|
| Modified | src/gridiron_edge/models/game_prediction/post_process.py |
| Modified | src/gridiron_edge/evaluation/diagnostics.py |
| Modified | src/gridiron_edge/evaluation/backfill.py |
| Modified | src/gridiron_edge/models/artifact.py |
| Modified | src/gridiron_edge/models/base.py |
| Modified | src/gridiron_edge/cli/evaluate.py |
| Modified | tests/unit/models/test_post_process.py |
| Modified | tests/unit/models/test_pipeline.py |
| Modified | tests/unit/market/test_recommendations.py |
| Modified | tests/integration/test_edges_cli.py |
| Modified | tests/integration/test_betting_cli.py |
| Modified | PLAN.md |
| Modified | HANDOFF.md |
| Added | scripts/recalibrate_sigma.py |
| Added | scripts/clean_archive.py |

##### Summary
- **2 new scripts**, 6 source files modified, 5 test files modified
- Resolves "Recalibrate sigma/margin_std after retrain" debt item

## 2026-06-04 - Champion/Challenger Model Refactor - Complete

Replaced versioned model variants with a champion/challenger system and
fixed temporal CV leakage in all model families.

#### Temporal CV fix
- `_features.py`: added chronological sort (`sort_values(["YEAR", "WEEK_NUM"])`)
  in `_prepare_data` so TimeSeriesSplit respects temporal ordering
- `tree.py`: switched RF and XGB from `StratifiedKFold(shuffle=True)` to
  `TimeSeriesSplit(n_splits=5)` for hyperparameter search CV
- `logistic.py`: switched `LogisticRegressionCV` from default 5-fold to
  explicit `TimeSeriesSplit(n_splits=5)` fold list
- `_features.py`: added `MIN_CV_TRAIN_ROWS = 4000` constant - early
  TimeSeriesSplit folds with <4000 rows are skipped during HP search
  to avoid undersized training sets biasing toward conservative HPs

#### Champion/challenger promotion system
- New module: `evaluation/champion.py`
  - `PromotionCriteria`: gate thresholds (Brier ≥ 0.002 improvement,
    ECE ≤ 0.01 degradation, AUC ≤ 0.01 degradation)
  - `ComparisonResult`: full comparison outcome with per-gate results
  - `compare_models()`: runs all gates, returns verdict
  - `format_comparison()`: human-readable metric table with ✅/❌ gates
  - `extract_metrics()`: standardised metric dict from ModelMetadata
- 13 unit tests (`tests/unit/evaluation/test_champion.py`)

#### Simplified model registry
- Replaced 10 versioned registrations with 3 unversioned champions:
  `random_forest`, `xgboost`, `logistic`
- Old versioned names (rf_v1–v3, xgb_v1–v3, logistic_v1–v4) removed
  from PredictorRegistry
- Versioned names retained only in `post_process.py` sigma/margin_std
  dicts for backward compatibility with old prediction archives
- Default model in `cli/edges.py` changed from `random_forest_v3` to
  `random_forest`
- Updated `diagnostics.py` model colors, `predictor.py` docstrings,
  `__init__.py` docstrings, `artifact.py` examples

#### CLI updates (`cli/models.py`)
- `gridiron models train <name>`: auto-compares challenger vs champion
  using promotion gates. First training auto-saves as champion.
  Backup/restore on rejection.
- `--force`: promote despite failed gates
- `--no-promote`: train and compare without replacing champion
- `gridiron models info <name>`: shows all 5 holdout metrics
- Removed `--overwrite` flag (replaced by auto-compare flow)

#### All training functions now store 5 holdout metrics
- Brier, ECE, AUC, log loss, accuracy stored in `parameters` dict
- RF: added `expected_calibration_error`, `roc_auc`, `log_loss`, `accuracy`
- XGB: added `roc_auc`, `log_loss`, `accuracy` (ECE already existed)
- Logistic: added all 4 (none existed previously)

#### Retrained champions (honest temporal CV metrics)

| Model | Brier | ECE | AUC | Accuracy | Notes |
|---|---|---|---|---|---|
| xgboost | 0.218 | 0.014 | 0.691 | 64.0% | 🏆 Auto-selected champion |
| random_forest | 0.220 | 0.013 | 0.702 | 64.3% | Best calibration |
| logistic | 0.225 | 0.017 | 0.683 | 63.5% | |
| elo_v2 (baseline) | 0.227 | 0.073 | 0.676 | 62.2% | All ML models beat Elo |

Note: metrics are lower than old rf_v3 (Brier 0.195, AUC 0.774) because
the old StratifiedKFold CV inflated HP selection. The new numbers are the
honest ones. Calibration (ECE) improved dramatically (0.036 → 0.013).

#### Files changed

| Action | File |
|---|---|
| Added | `src/gridiron_edge/evaluation/champion.py` |
| Modified | `src/gridiron_edge/models/game_prediction/tree.py` |
| Modified | `src/gridiron_edge/models/game_prediction/logistic.py` |
| Modified | `src/gridiron_edge/models/game_prediction/_features.py` |
| Modified | `src/gridiron_edge/models/game_prediction/post_process.py` |
| Modified | `src/gridiron_edge/models/game_prediction/predictor.py` |
| Modified | `src/gridiron_edge/models/game_prediction/__init__.py` |
| Modified | `src/gridiron_edge/models/artifact.py` |
| Modified | `src/gridiron_edge/cli/models.py` |
| Modified | `src/gridiron_edge/cli/edges.py` |
| Modified | `src/gridiron_edge/evaluation/diagnostics.py` |
| Added | `tests/unit/evaluation/test_champion.py` |
| Modified | `tests/unit/models/test_tree_models.py` |
| Modified | `tests/integration/test_edges_cli.py` |
| Modified | `tests/unit/market/test_recommendations.py` |

#### Summary
- **1 new source file**, 10 modified
- **1 new test file**, 3 modified, **13 new tests**
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-03 - W6: Portfolio & Bet Tracking - Complete

The feedback loop - track bets, measure performance, prove (or disprove)
the system works.  The M2 milestone.  Builds on W5 (edge context for
bets), W3 (market math for PnL), and W1 (odds ledger for CLV on
settlement).

#### Bet ledger (`betting/ledger.py`)
- Append-only Parquet ledger following the `archive.py` pattern
- 20-column schema: bet context (game, market, side, odds, stake, book),
  model context (version, prob, EV, strength, tier), settlement
  (status, settled_at, pnl, closing_line, closing_odds, clv)
- `compute_pnl()`: pure function - won = stake × (decimal_odds − 1),
  lost = −stake, push/open = 0
- `log_bet()`: generate UUID, append row with status "open", return bet_id
- `settle_bet()`: validate open, compute PnL, optionally compute CLV
  from odds ledger (ML = probability-based, spread/total = point-based)
- `load_bets()`: load with filters (status, season, week, market_type, book)
- Fixed pandas FutureWarning: `dropna(axis=1, how="all")` + `reindex` for concat
- Fixed pandas FutureWarning: `pd.to_datetime()` cast before `settled_at` assignment
- 24 unit tests (`tests/unit/betting/test_ledger.py`)

#### Bankroll management (`betting/bankroll.py`)
- Decoupled from ledger - CLI orchestrates both
- Transaction types: deposit, withdraw, bet_placed, bet_settled
- Sign convention: deposits/settlements = positive, withdrawals/bets = negative
- `deposit()` / `withdraw()`: record cash movements (positive amounts only)
- `record_bet_placed(stake)`: record stake leaving bankroll
- `record_bet_settled(stake, pnl)`: record gross return (stake + pnl)
  - won: stake + profit, lost: 0, push: stake
- `current_balance()`: sum of all signed transactions
- `balance_history()`: running balance DataFrame with cumulative sum
- `load_transactions()`: load with optional txn_type filter
- Same `dropna` + `reindex` concat pattern as ledger
- 23 unit tests (`tests/unit/betting/test_bankroll.py`)

#### Performance analytics (`betting/performance.py`)
- Pure DataFrame-in, results-out - no I/O
- `record()`: W-L-P counts, win_pct (pushes excluded from denominator),
  optional `split_by` for grouping
- `roi()`: total_staked, total_pnl, roi_pct, optional `split_by`
- `clv_summary()`: mean/median CLV, % positive, n_bets
- `ev_analysis()`: mean_ev_at_bet, mean_actual_roi, ev_vs_actual_gap
- `streak_analysis()`: current streak (±), longest W/L streaks,
  pushes break streaks
- `summary()`: combined dashboard dict calling all of the above
- Kelly adherence deferred (requires `recommended_stake` in ledger schema)
- 22 unit tests (`tests/unit/betting/test_performance.py`)

#### CLI (`cli/betting.py`)
- 8 commands registered as `gridiron bet` in `cli/main.py`
- `gridiron bet log`: record bet → `log_bet()` + `record_bet_placed()`
- `gridiron bet settle <id> <result>`: settle → `settle_bet()` +
  `record_bet_settled()`, optional CLV via `--with-clv/--no-clv`
- `gridiron bet list`: show bets with optional status/market filters
- `gridiron bet summary`: performance dashboard with optional `--split-by`
- `gridiron bet balance`: current balance + recent transaction history
- `gridiron bet export`: CSV export to `data/output/bets/`
- `gridiron bet deposit <amount>`: add funds
- `gridiron bet withdraw <amount>`: remove funds
- Graceful error handling throughout (not found, already settled, invalid amount)
- 17 integration tests (`tests/integration/test_betting_cli.py`)

#### Manual validation
- Full round-trip verified: deposit → log → list → settle → summary →
  balance → export → withdraw
- Math verified: deposit $1000, bet $100 at −150, won → PnL +$66.67,
  balance $1066.67. Second bet $50 spread, lost → balance $1016.67.
  Withdraw $200 → balance $816.67. All correct.

#### Files changed
| Action | File |
|---|---|
| Added | `src/gridiron_edge/betting/__init__.py` |
| Added | `src/gridiron_edge/betting/ledger.py` |
| Added | `src/gridiron_edge/betting/bankroll.py` |
| Added | `src/gridiron_edge/betting/performance.py` |
| Added | `src/gridiron_edge/cli/betting.py` |
| Modified | `src/gridiron_edge/cli/main.py` (import + register `betting_app`) |
| Added | `tests/unit/betting/__init__.py` |
| Added | `tests/unit/betting/test_ledger.py` |
| Added | `tests/unit/betting/test_bankroll.py` |
| Added | `tests/unit/betting/test_performance.py` |
| Added | `tests/integration/test_betting_cli.py` |

#### Summary
- **4 new source files**, 1 modified
- **4 new test files**, **86 new tests** (24 + 23 + 22 + 17)
- `betting/` package: 3 modules (ledger, bankroll, performance)
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-02 - W5: Edge Engine - Complete

The convergence point - model predictions meet market prices to surface
betting edges.  Builds on W1 (odds ingest & joins), W2 (enriched
predictions with spreads/bands/tiers), and W3 (market math in
odds_math/kelly).

#### Edge calculation core (`market/edge.py`)
- Pure scalar functions, no I/O - follows the `odds_math.py` / `kelly.py` leaf pattern
- 3 frozen dataclasses: `MoneylineEdge`, `SpreadEdge`, `TotalEdge`
- `expected_value()`: EV = model_prob * decimal_odds - 1
- `moneyline_edge()`: no-vig debiases market, returns +EV side or None
- `spread_cover_prob()`: probit P(home covers) via calibrated `margin_std`
- `spread_edge()`: cover prob -> EV -> Kelly -> +EV side or None
- `total_cover_prob()`: probit P(over) via total model residual std
- `total_edge()`: over/under prob -> EV -> Kelly -> +EV side or None
- `classify_edge_strength()`: EV -> strong (>=5%) / moderate (2-5%) / lean (0-2%) / no_edge
- 37 unit tests (`tests/unit/market/test_edge.py`)

#### Edge report builder (`market/recommendations.py`)
- `pivot_odds_to_wide()`: long-format odds -> one row per game (handles duplicate fetches via groupby/last)
- `join_predictions_to_odds()`: inner-join predictions <-> wide odds on `game_id` (auto-pivots long odds)
- `compute_game_edges()`: single game -> list of edges across all available markets, graceful NaN handling
- `build_edge_report()`: full orchestrator -> 18-column report DataFrame
  - Kelly stake = bankroll * kelly_multiplier * kelly_frac (capped at bankroll * kelly_multiplier)
  - `classify_edge_strength()` applied to every row
- `rank_edges()`: filter to `ev > min_ev`, sort descending
- 21 unit tests (`tests/unit/market/test_recommendations.py`)

#### Closing Line Value (`market/clv.py`)
- `closing_line_value()`: probability-based CLV = (close_prob - bet_prob) / bet_prob
- `spread_clv()`: point-based CLV for spread bets (home: bet - close; away: close - bet)
- `total_clv()`: point-based CLV for total bets (over: close - bet; under: bet - close)
- `extract_opening_odds()` / `extract_closing_odds()`: first / last pull per (game_id, market, side) from ledger
- `build_clv_report()`: augments edge report with `opening_value`, `closing_value`, `clv` columns
- `summarize_clv()`: mean, median, pct positive, edge count
- Reuses `pivot_odds_to_wide` from `recommendations.py` via `_pivot_and_suffix()` (DRY)
- 30 unit tests (`tests/unit/market/test_clv.py`)

#### CLI (`cli/edges.py`)
- `gridiron edges report --week N --season YYYY-YYYY`
  - Loads prediction archive + current odds -> builds edge report -> ranks by EV
  - Rich console table: color-coded EV (green/yellow/dim), Kelly stakes, confidence tiers
  - CSV export via `--format csv` to `data/output/edges/`
  - Options: `--model-version`, `--bankroll`, `--kelly-multiplier`, `--min-ev`
- `gridiron edges clv --season YYYY-YYYY`
  - Loads predictions + full odds ledger -> builds edge report -> computes CLV -> summary stats
- Graceful empty-data handling throughout (no predictions, no odds, no edges)
- Registered in `cli/main.py` as `edges_app`
- 6 integration tests (`tests/integration/test_edges_cli.py`)

#### Files changed
| Action | File |
|---|---|
| Added | `src/gridiron_edge/market/edge.py` |
| Added | `src/gridiron_edge/market/recommendations.py` |
| Added | `src/gridiron_edge/market/clv.py` |
| Added | `src/gridiron_edge/cli/edges.py` |
| Modified | `src/gridiron_edge/cli/main.py` (import + register `edges_app`) |
| Modified | `src/gridiron_edge/market/__init__.py` (re-exports) |
| Added | `tests/unit/market/test_edge.py` |
| Added | `tests/unit/market/test_recommendations.py` |
| Added | `tests/unit/market/test_clv.py` |
| Added | `tests/integration/test_edges_cli.py` |

#### Summary
- **4 new source files**, 2 modified
- **4 new test files**, **94 new tests** (37 + 21 + 30 + 6)
- `market/` package: 5 modules (odds_math, kelly, edge, recommendations, clv)
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-02 - W2: Richer Game Model Outputs - Complete

Extended game prediction models to produce spread, total, projected scores,
uncertainty bands, and confidence tiers - not just win probability.

#### Post-processing enrichment (`post_process.py`)
- **Spread derivation:** probit link with per-model sigma calibration (13 variants)
  - Best: random_forest_v3 (sigma=13.97, spread MAE vs Vegas=3.16, r=0.80)
- **Isotonic recalibration:** infrastructure built, decision gate rejected for rf_v3
  (holdout ECE 0.036 already excellent; recalibration worsened it to 0.083)
- **Uncertainty bands:** 90% credible intervals via spread ± z*margin_std → probit
  - Per-model margin_std registry (best: rf_v3 at 12.85, worst: elo_v1 at 13.89)
- **Confidence tiers:** band width → High (<0.65) / Moderate (0.65–0.82) / Low (≥0.82)
  - Validated: High 96.8%, Moderate 86.8%, Low 64.0% favored-team win rate
- **Projected scores:** home = (total - spread) / 2, away = (total + spread) / 2
  - Home MAE: 6.95, Away MAE: 6.74, near-zero bias

#### Total points model (`total.py`)
- Random Forest regressor targeting actual_total = PTS_WINNER + PTS_LOSER
- Uses same 107-feature expanded set as win models
- TimeSeriesSplit CV (not KFold) to avoid temporal leakage
- total_rf_v1 trained: holdout MAE=10.27, RMSE=13.17 (n=1,467)
- Competitive with Vegas closing totals (model MAE 3.11 vs closing O/U, r=0.64)

#### Prediction pipeline (`pipeline.py`)
- Composable orchestrator: load → predict (win) → predict (total) → build rows → enrich
- `predict_games()` replaces monolithic `_predict_historical_tree()` internals
- `build_game_predictions()` maps raw model output to game-level rows
- All model families (elo, logistic, tree) now produce enriched predictions

#### Archive schema extension (`archive.py`)
- 8 new columns: model_spread, model_total, projected_home_score,
  projected_away_score, margin_std, win_prob_lo, win_prob_hi, confidence_tier
- Backward compatible: old archives load with NaN fill for missing columns

#### Validation report (rf_v3 vs Vegas)
| Metric | Value |
|--------|-------|
| Spread MAE vs closing line | 3.16 |
| Spread correlation | 0.80 |
| Total MAE vs closing O/U | 3.11 |
| Total correlation | 0.64 |
| Home score MAE | 6.95 |
| Away score MAE | 6.74 |
| High confidence fav win% | 96.8% |
| Moderate confidence fav win% | 86.8% |
| Low confidence fav win% | 64.0% |

**Note:** VEGAS_LINE uses opposite sign convention from model_spread
(positive = home favored vs negative = home favored). Documented in HANDOFF.md.

#### Phase reference cleanup
Scrubbed all Phase A/B/C/D/E/20c/20d/20e and W2 references from source and
test files. Replaced with descriptive terminology. PLAN.md and CHANGELOG.md
retain historical phase references since they are historical records.

#### Tests added: 44 new (total ~456)
- test_post_process.py: 33 → 55 (bands, tiers, enrichment)
- test_total.py: 11 (projected scores, enrichment with total)
- test_pipeline.py: 7 (build_game_predictions)
- test_archive_schema.py: 4 (schema extension, backward compat)

## 2026-06-01 - Phase 20e Feature Engineering Complete

Completed Priorities 1-7 + 14-15 across three batches:
- Batch 1: Rest differential + explosive play rate (+8 columns)
- Batch 3: PBP efficiency (success splits, 3rd down, red zone,
  turnovers, sack rate) (+36 columns)
- Batch 2/15: Weather & venue wiring verified already complete

Feature count: _EXPANDED_FEATURES 16 -> 107. EPA_COLS 8 -> 22.
Model features now cover EPA, efficiency splits, explosiveness,
situational football, turnovers, pass rush, rest, weather, venue.

Remaining Phase 20e backlog: Priorities 8-13 (CPOE, pace, score
differential, penalties, special teams, coaching). These require
additional PBP columns or external data sources.

Next active workstream: W2 (Richer Game Model Outputs).

## W3: Market Intelligence Foundation - 2026-05-31

### New package: `market/`
- Pure-math leaf package at `src/gridiron_edge/market/` - no data dependencies,
  no pandas, no I/O

### `market/odds_math.py`
- `american_to_decimal()`: American → decimal odds conversion
- `american_to_implied_prob()`: American → raw implied probability (includes vig)
- `decimal_to_american()`: decimal → American; even-money normalises to +100
- `hold_pct()`: bookmaker overround for two-way markets
- `no_vig()`: fair probabilities via power method (default) or additive rescaling
- `_power_devig()`: bisection solver for `raw_a^k + raw_b^k = 1` - no scipy

### `market/kelly.py`
- `kelly_fraction()`: full-Kelly optimal fraction; returns 0 when edge ≤ 0
- `kelly_stake()`: dollar amount using fractional Kelly (default quarter-Kelly)
- Input validation: probability must be in (0, 1), bankroll ≥ 0, fraction in [0, 1]

### Tests added (64)
- `test_odds_math.py` (42) - conversions, roundtrips, extreme odds (±10000),
  hold percentage, no-vig additive vs power, sums-to-one, fair-probs-not-above-raw
- `test_kelly.py` (22) - positive/negative/zero edge, fractional staking,
  zero bankroll, guard rails on probability/bankroll/fraction

### Deferred
- `market/consensus.py` - deferred until multi-book data available (W7)

## W1: Quick Wins & Unblocking - 2026-05-31

### DK Unicode Minus Fix
- `ingest/odds/draftkings.py` → `_norm_display_odds_american()`: handle Unicode
  minus (U+2212) before `isdigit()` check and `int()` conversion. DraftKings API
  returns `"−150"` with U+2212 instead of ASCII hyphen; this caused `ValueError`
  on all negative odds parsing.

### DK `game_id` Resolver
- New module: `ingest/odds/_game_id.py`
- `team_long_to_short()`: reverse lookup from `NFLVERSE_SHORT_TO_LONG`, with
  historical relocation codes (`OAK`, `SD`, `STL`) deprioritized so current
  codes (`LV`, `LAC`, `LA`) always win
- `build_game_id()`: constructs canonical `YYYY_WW_AWAY_HOME` format
- `resolve_dk_game_ids()`: vectorized column addition supporting both
  intermediate (`home_team`/`away_team`) and wide (`team`/`opponent`/`location`)
  DataFrame formats

### End-to-End Odds Join Validation
- Integration test confirms predictions ↔ odds join on `game_id` at 100% match
  rate on synthetic data, with left-join surfacing unmatched games as nulls

### Tests added (25)
- `test_draftkings_parse.py` (9) - Unicode minus, positive, int/float passthrough,
  fallback keys, non-numeric string, missing keys
- `test_game_id.py` (13) - team lookup, all 32 teams resolve, build_game_id format,
  week padding, unknown teams → None, both DataFrame formats, column preservation
- `test_odds_join.py` (3) - canonical format validation, inner join match rate,
  left join null surfacing


## W0 Complete: Test Framework Build-Out - 2026-05-31

### Summary
Professional three-tier testing infrastructure (unit → integration → e2e)
with automated quality gates, shared fixtures, and 412 tests at 40% coverage.

### Phases completed
- **Phase 0** - Foundation: directory restructure, auto-markers, shared fixtures,
  pre-commit/pre-push hooks, coverage config
- **Phase 1** - Core & Datasets: 60 tests covering constants, paths, settings,
  registry, loaders, writers, accessor
- **Phase 2** - Missing Features: 63 tests covering all 11 feature modules,
  feature registry, FeatureSpec protocol
- **Phase 3** - Models & Evaluation: 35 tests covering Predictor/Trainable
  protocols, model registry, artifact store, backfill, select, tune, diagnostics
- **Phase 4** - Ingest, Transform, Sim: 65 tests covering odds store, nflverse
  helpers, sim types/constants, geo/haversine, DK fixture validation
- **Phase 5** - Integration & E2E: 28 tests covering dataset roundtrips,
  artifact roundtrips, CLI workflows, full prediction pipeline via MiniRepoBuilder
- **Deferred resolution** - Added test_tune.py (16 tests), test_diagnostics.py
  (8 tests), removed slow training tests that exercised sklearn/xgboost internals

### Coverage baseline
- 412 tests, 0 failed, 0 deselected
- 40.04% line coverage (threshold: 40%, ratchet up over time)
- Core business logic (features, datasets, evaluation) at 80-100%
- Sim, viz, CLI, and model training code deferred to respective workstreams

### Deferred test areas (to be added with respective workstreams)
- Numba sim kernels: `test_engine.py`, `test_playoffs.py` (sim workstream)
- DK API mocking: full `test_draftkings.py` (odds workstream)
- Elo predictor: `test_elo_predictor.py` (elo workstream)
- Transform ETL: `test_epa_transform.py` (data pipeline workstream)
- Cosmetic: migrate inline imports → top-level; migrate local helpers → shared fixtures


### Test Framework Build-Out - 2026-05-31

Established professional three-tier testing infrastructure.

**Test directory restructure**
- Restructured `tests/` into `unit/`, `integration/`, `e2e/` subdirectories
- Tests auto-tagged by directory via `pytest_collection_modifyitems` in root conftest - no manual `@pytest.mark` decorators needed
- Existing tests moved to `tests/unit/` with zero import changes required

**Shared fixtures**
- `tests/fixtures/dataframes.py` - 9 centralized DataFrame factories: `make_games`, `make_modeling_rows`, `make_stadiums`, `make_elo_state`, `make_epa_by_game`, `make_weather_enriched`, `make_eval_df`, `make_predictions`, `make_accessor`
- `tests/fixtures/repos.py` - composable `MiniRepoBuilder` class (builder pattern: `.with_games().with_stadiums().with_elo_state().build()`)
- Replaces duplicated `_make_games()`, `_make_eval_df()`, `mini_repo` patterns across 8+ test files

**Pre-commit / pre-push hooks:**
- Added `.pre-commit-config.yaml` with two stages:
  - `pre-commit`: ruff lint + format, pyrefly type check, unit tests
  - `pre-push`: integration + e2e tests
- Installed via `pre-commit install` + `pre-commit install --hook-type pre-push`
- Safety valve: `|| test $? -eq 5` allows commits during incremental marker migration

**Pytest configuration:**
- Added markers to `pyproject.toml`: `unit`, `integration`, `e2e`, `slow`, `network`
- `--strict-markers` enforced - no typos in marker names
- Coverage config added: `fail_under = 60`, `show_missing = true`

**Fixed drifted tests**
- `test_home_field_feature`: `GAME_LOCATION` `"NULL_VALUE"` → `"H"` (aligned with constants consolidation)
- `test_weather`: `_make_modeling_row` returns DataFrame not dict; `test_null_value_string_gives_nan` assertion updated
- `test_tree_models`: imports updated for `_epa_window` module extraction (`_rebuild_features_with_window`, `_EPA_WINDOW_OPTIONS`)
- `test_features_pipeline`: `pd.read_csv` → `pd.read_parquet` for `modeling_base`/`modeling_full`
- Model training tests (`TestRandomForestV1Training`, `TestXGBoostV1Training`) marked `@pytest.mark.slow` (~15min each)

**Tooling**
- `mirror_repo_to_sharepoint.py` - mirrors repo to SharePoint-synced folder for Copilot indexing. Copies `.py` files as `.py.txt` with SOURCE headers; preserves `.md`/`.json`/`.yaml` as-is. Supports `--clean`, `--dry-run`, `--extra-ext`.


## Thermonuclear Code Quality Review - 2026-05-30

Eight review batches across the full codebase, followed by six implementation passes and full pipeline validation. All changes committed in four atomic commits.

### Pass 1+2 - Constants consolidation + Elo engine

**Constants - single source of truth in `core/constants.py`:**
- `HOME_GAME_LOCATION = "H"`, `AWAY_WIN_LOCATION = "@"`, `HOLDOUT_SEASONS`, `EXPANSION_TEAMS` - all previously defined independently in 2–4 files each
- Retired the PFR-era `"NULL_VALUE"` home-game sentinel → `"H"` for `GAME_LOCATION`; `""` for all missing data fields (GAMETIME, STADIUM, ROOF, SURFACE, GAME_DATE, GAME_DAY_OF_WEEK) across the transform layer
- All consumers updated: `venue_hfa`, `home_field`, `record`, `primetime`, `backfill`, `tune`, `elo/predictor`, `metrics`, `schedule_nflverse`, `games_nflverse`, `_nflverse_common`
- Deleted dead placeholder packages: `datasets/contracts/`, `analytics/`, `config/`

**Elo engine - parameterised divisor:**
- `ratings/elo/core.py`: `elo_win_probability(divisor=DEFAULT_ELO_DIVISOR)` and `update_elo(divisor=)` - divisor no longer hardcoded to 480
- `EloTableConfig` gains `divisor: float = 480.0`; `_build_elo_dict` passes it through
- `tune.py`: `_win_prob` deleted - `_simulate_and_score` delegates to `core.elo_win_probability`
- `SimulationConfig` gains `divisor: float = 480.0`; numba `_elo_win_prob`/`_elo_update` in `sim/_engine.py` accept divisor as a parameter
- `gridiron sim run` gains `--divisor` flag

### Batch 1-8 code review fixes

Individual file-level fixes from all 8 review batches:
- `DatasetSpec`: dropped redundant `key` field (14 instantiations updated)
- `FeatureRegistry`: duplicate-name guard + descriptive `KeyError` in `register()`/`get()`
- `features/team/epa.py`: vectorised inner EPA rolling loop; extracted `_join_team_epa` helper; `EPA_COLS` made public
- `ratings/elo/table.py`: deleted backwards-compat alias `update_elo_state_table_incremental`
- `evaluation/diagnostics.py`: filled `_MODEL_COLORS` gaps for logistic_v4, random_forest_v1/v2, xgboost_v2
- `evaluation/metrics.py`: removed duplicate `_archive_path` and `load_prediction_log` - now imports from `archive.py`
- `viz/excel.py` → `viz/rankings.py`: renamed; `cli/output.py` updated
- `metrics/travel/geo.py`: `Tude` type alias renamed to `CoordinateValue`
- `backfill.py`, `tune.py`, `metrics.py`: local `_AWAY_WIN_LOCATION` definitions removed, imported from `core.constants`

### Pass 3 - File decomposition

**`sim/season.py`** (1235 lines) split into three files:
- `sim/_types.py` - constants, all config dataclasses (`SimulationConfig`, `SimPaths`, `TeamIndex`, `ScheduleArrays`, `SimulationResults`), `_log_phase`, `format_record`. Pure-data leaf - no I/O, no numba.
- `sim/_engine.py` - numba kernels: `_elo_win_prob`, `_elo_update`, `apply_actuals_to_matrices`, `simulate_remaining_regular_season`, `precompute_game_counts`
- `sim/season.py` - data loading, output builders, `run_full_simulation` (~734 lines)
- `sim/__init__.py` - public API re-exports; sync assertions validate `playoffs.py` constants match `_types.py` at import time
- `viz/charts.py` - import updated from `sim.season` → `sim._types`

**`models/game_prediction/_shared.py`** (333 lines) split:
- `_columns.py` - schema version, all column lists, `FeatureSet` dataclass; pure-data leaf
- `_features.py` - feature engineering functions, `FEATURE_SETS` dict, `_prepare_data`, `_is_trained`
- `_shared.py` - thin re-export shim (33 lines)
- `logistic.py` and `tree.py` updated to import from new modules directly

**`models/game_prediction/tree.py`** (984 lines):
- `_epa_window.py` extracted - `_EPA_RAW_COLS`, `_EPA_COL_MAP`, `_EPA_WINDOW_OPTIONS`, `WindowData` NamedTuple, `_rebuild_features_with_window`, `_get_cached_window_data`
- `tree.py` reduced to 820 lines

**Final line counts:** no file exceeds 820 lines. `playoffs.py` ↔ `_types.py` constant sync is machine-checked at import time.

### Pass 4 - Feature dependency enforcement

- `features/base.py`: `FeatureSpec` gains `depends_on: Sequence[str] = ()` field
- `features/registry.py`: `validate_ordering(feature_names)` - raises `ValueError` at import time if ordering violates any `depends_on` constraint
- `features/pipeline.py`: calls `validate_ordering(FEATURES)` at module level
- Dependencies declared: `travel` → `home_field`; `venue_hfa` → `travel`; `schedule_strength` → `team_elo`

### Pass 5 - CLI stage-list pattern

- `cli/main.py`: 10 boolean flags replaced with `--skip STAGE` / `--only STAGE` repeatable options
- `ALL_STAGES` defines the canonical stage vocabulary: `fetch-games`, `clean-games`, `fetch-upcoming`, `clean-upcoming`, `fetch-weather`, `fetch-odds`, `build-epa`, `build-elo`, `build-features`
- Dead `build-epa` stage fixed - was declared but never executed
- `PLR0912`/`PLR0915` suppressions moved to `_run_pipeline_stages` where they belong; `run_data_pipeline` is now clean
- `evaluation/select.py` introduced - `collect_model_metrics`, `rank_models`, `compute_report_data` extracted from `cli/evaluate.py`

### Pass 6 - Archive schema migration

- `evaluation/archive.py`: `is_backfilled: bool` column added to schema; `build_archive_rows` and `append_to_prediction_log` gain `is_backfilled` parameter; `write_archive_rows` and `load_prediction_log` backward-compatible; `migrate_archive()` added
- `models/elo/predictor.py`: `_BACKFILL_TS` constant deleted; predictions use actual timestamp + `is_backfilled=True`
- `logistic.py`, `tree.py`: inline `datetime(1970, 1, 1)` sentinels replaced with actual timestamp + `is_backfilled=True`

### Post-commit fixes

- `ingest/weather/openweather.py` - `fetch_weather` now reads existing `weather_enriched.csv` and fetches only games not already enriched. Idempotent - safe to re-run without duplicating rows.
- `sim/season.py` - `run_full_simulation` raises `FileNotFoundError` with actionable message when the upcoming schedule CSV is empty, instead of a cryptic `IndexError`.

---

## Phase 20d - Tree-based models (RF + XGBoost)

- `models/game_prediction/tree.py` - Random Forest and XGBoost variants registered alongside logistic models
- `models/game_prediction/logistic.py` - v3 and v4 logistic variants added
- `PredictorRegistry` - `register` + `get` + `trainable_names()` pattern generalised
- `evaluation/tune.py` - hyperparameter grid search for Elo K/divisor and EPA window
- `evaluation/diagnostics.py` - calibration plots, model comparison charts

---

## Phase 20c - Model reporting

- `evaluation/select.py` - `select_model` + `generate_report` pipeline
- `cli/evaluate.py` - `evaluate report`, `evaluate select-model`, `evaluate calibration` commands
- Full model characterisation: Brier score, log loss, calibration, accuracy per season

---

## Phase 20b - Model evaluation infrastructure

- `evaluation/metrics.py` - Brier score, log loss, calibration table, accuracy
- `evaluation/backfill.py` - `backfill_model(model_version)` covering all registered models
- `evaluation/archive.py` - append-only prediction log at `predictions_log.parquet`
- `cli/evaluate.py` - `evaluate backfill`, `evaluate summary` commands

---

## Phase 20a - Prediction engine

- `models/game_prediction/logistic.py` - logistic v1 + v2 registered predictors
- `models/base.py` - `Predictor` + `Trainable` protocols
- `models/registry.py` - `PredictorRegistry`
- `models/artifact.py` - `ArtifactStore` (joblib-based)
- `cli/models.py` - `models train`, `models list` commands

---

## Phase 19 - Football state representation (EPA, rest, travel, records)

- `features/team/epa.py` - rolling EPA features from PBP data
- `features/team/rest.py` - days rest, short week, post-bye flags
- `features/team/travel.py` - km traveled, timezone shift
- `features/team/record.py` - win/loss/tie record, win streak
- `features/team/schedule_strength.py` - SOS, SOV
- `ingest/nflverse/pbp.py` - play-by-play ingestion
- `transform/clean/epa.py` - PBP → game-level EPA aggregation
- Schema v3 modeling file with all Phase 19 features

---

## Phase 18 - Evaluation infrastructure

- Prediction archive - append-only Parquet log
- `evaluation/metrics.py` - Brier score, log loss, calibration, accuracy
- `evaluation/backfill.py` - generic backfill covering all registered models
- `evaluation/tune.py` - Elo parameter grid search
- `datasets/manifest.py` - schema versioning for modeling files

---

## Phase 15-17 - Excel retirement, Scrapy retirement, dead code removal

- `ingest/odds/` - DraftKings odds ingest + append-only Parquet ledger
- `ingest/odds/store.py` - long-format odds storage with dedup
- `viz/predictions.py` - weekly matchup PNG + static HTML (migrated from notebook)
- `viz/rankings.py` - Elo rankings CSV (was Excel)
- Scrapy / PFR scraper fully deleted
- Dead stub files removed; all ruff/pyrefly gates passing

---

## Phase 13-14 - nflverse migration + console system

- Replaced PFR/Scrapy with `nfl_data_py` - bypasses Cloudflare
- `ingest/nflverse/` - game + schedule + upcoming ingestion
- `transform/clean/games_nflverse.py` + `schedule_nflverse.py` - canonical schema mappers
- `core/console.py` - timed step context manager, header/summary banners, verbose mode
- `core/logging.py` - WARNING in compact mode, DEBUG in verbose

---

## Phases 1-12 - Core refactor + tooling

Original migration from `data_pipelines/` + `model_pipelines/` + `utils/` into `src/gridiron_edge/`. uv migration, Ruff + Pyrefly quality gates, Google-style docstrings, full type annotation pass. See git history for full detail.

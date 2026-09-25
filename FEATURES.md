# Gridiron Edge - Feature Catalog & Modeling Reference

## Every model feature (existing, planned, aspirational) plus the methods, data, and validation practices behind them

---

## How This Document Fits

| Document | Relationship |
|----------|-------------|
| **FEATURES.md** (this file) | Living reference of all features across all domains (Part I), the methods and validation practices behind them (Part II), and the feature priority queue (Part III). Updated as features are built, added, or deprioritized. |
| **ROADMAP.md** | References feature domains at the workstream level. Links here for detail. |
| **PLAN.md** | Pulls specific features from the Priority Matrix into short-term task lists. |
| **CHANGELOG.md** | Records when features move from Missing to Done. |

Domains 1–11 keep their original numbers so existing references still resolve. Domains 12–14 are new in the 2026-09-24 revision.

### Column definitions

| Column | Meaning |
|--------|---------|
| **Model Target** | Which model(s) this feature feeds: **Game** (M2), **Props** (M3), **What-If** (M4/scenario), **Market** (M5) |
| **Signal** | Estimated predictive value: 🔴 High / 🟡 Med / 🟢 Low / ❓ Unknown |
| **Cost** | Implementation effort: Low (data exists, simple transform) / Med / High (new data source or complex engineering) |
| **Status** | ✅ Done / ⚠️ Partial / ❌ Missing |

### Row markers

| Marker | Meaning |
|--------|---------|
| 🆕 | Added in the 2026-09-24 revision. A proposal: signal and cost are estimates, not yet checked against the codebase. |
| ↺ | Row added for a feature that is already built but had no catalog row. Confirm against the code. |

Rows without a marker carry over from the 2026-06-10 revision. Any description or cost edits to them are itemized in the Changelog.

---

# Part I — Feature Catalog

## Domain 1: Team Offensive Efficiency

*What does this team's offense actually do, and how well?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| EPA/play (overall) | Expected points added per play, rolling window | Game, Props | 🔴 High | Low | ✅ Done |
| EPA/play (pass) | Passing EPA per dropback | Game, Props | 🔴 High | Low | ✅ Done |
| EPA/play (rush) | Rushing EPA per carry | Game, Props | 🔴 High | Low | ✅ Done |
| Success rate (overall) | % of plays gaining positive EPA | Game | 🟡 Med | Low | ✅ Done |
| Success rate (pass/rush split) | Passing vs rushing success rate | Game, Props | 🟡 Med | Low | ✅ Done |
| Explosive play rate | % of plays gaining 20+ yds (pass) or 10+ yds (rush) | Game | 🟡 Med | Low | ✅ Done |
| Scoring rate | Points per drive | Game | 🟡 Med | Low | ❌ |
| Red zone TD % | TD rate when inside opponent 20 | Game | 🟡 Med | Low | ✅ Done |
| Red zone attempts/game | Volume of red zone trips | Game, Props | 🟡 Med | Low | ✅ Done |
| 3rd down conversion % | Overall and by distance bucket | Game | 🟡 Med | Low | ✅ Done |
| Plays per game / pace | Tempo proxy - affects volume stats | Game, Props | 🟡 Med | Low | ✅ Done |
| Time of possession | Average TOP | Game | 🟢 Low | Low | ❌ |
| Pass rate (neutral script) | Pass-heavy when game is close? Build as pass rate over expected (PROE): nflfastR PBP already carries `xpass` and `pass_oe`, so no new model is needed | Props | 🟡 Med | Low | ❌ |
| Pass rate (overall) | Raw pass/run ratio | Props | 🟢 Low | Low | ❌ |
| Yards per play | Simpler efficiency proxy | Game | 🟢 Low | Low | ✅ Done |
| CPOE | Completion % over expected. Per-play `cpoe` is already in PBP; the NaN exclusion is usually fixed with a NaN-aware mean over valid attempts plus an early-season prior (Part II.7) | Game, Props | 🔴 High | Low | ⚠️ Partial |
| Air yards / attempt | Depth of target proxy (`air_yards` is in PBP) | Props | 🟡 Med | Low | ❌ |
| YAC / completion | Yards after catch - scheme/personnel signal (`yards_after_catch` is in PBP) | Props | 🟡 Med | Low | ❌ |
| 🆕 Opponent-adjusted EPA/play (pass, rush) | Each offense's EPA with the strength of the defenses it faced removed: ridge or mixed-effects regression on play-level EPA with offense and defense terms. The DIY stand-in for DVOA (Part II.1) | Game, Props | 🔴 High | Med | ❌ |

---

## Domain 2: Team Defensive Efficiency

*How well does this team prevent the opponent from doing things?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Def EPA/play (overall) | Defensive expected points allowed per play | Game, Props | 🔴 High | Low | ✅ Done |
| Def EPA/play (pass) | Against the pass | Game, Props | 🔴 High | Low | ✅ Done |
| Def EPA/play (rush) | Against the run | Game, Props | 🔴 High | Low | ✅ Done |
| Def success rate | % of opponent plays held to negative EPA | Game | 🟡 Med | Low | ✅ Done |
| Pressure rate | QB pressures / dropbacks. No manual charting needed: nflverse participation carries a per-play `was_pressure` flag (recent seasons; published only after each season ends), and PFR advanced stats (2018+) are the likely in-season source. Calibrate one to the other before mixing (Part II.7) | Game, Props | 🔴 High | Med | ❌ |
| Sack rate | Sacks / dropbacks | Game, Props | 🟡 Med | Low | ✅ Done |
| Rush yards allowed / game | Volume stat, useful for prop matchups | Props | 🟡 Med | Low | ❌ |
| Pass yards allowed / game | Volume stat | Props | 🟡 Med | Low | ❌ |
| Opponent 3rd down conversion % | Defensive 3rd down stops | Game | 🟡 Med | Low | ✅ Done |
| Opponent red zone TD % | Bending but not breaking? | Game | 🟡 Med | Low | ✅ Done |
| Explosive plays allowed rate | Big play vulnerability | Game | 🟡 Med | Low | ✅ Done |
| Turnover creation rate | Forced fumbles + INTs per game | Game | 🟡 Med | Low | ✅ Done |
| Opponent completion % | Raw passing defense | Props | 🟢 Low | Low | ❌ |
| Def DVOA (if sourced) | Opponent-adjusted efficiency, now published by FTN (Football Outsiders closed in 2023); paywalled. The DIY opponent-adjusted row below covers most of the idea | Game | 🔴 High | High | ❌ |
| Points allowed / game | Simple but noisy | Game | 🟢 Low | Low | ❌ |
| Opponent QB rating | Passer rating allowed | Props | 🟡 Med | Med | ❌ |
| 🆕 Opponent-adjusted def EPA/play (pass, rush) | Defensive mirror of the Domain 1 row, fit in the same regression | Game, Props | 🔴 High | Med | ❌ |

---

## Domain 3: Turnover & Discipline

*Variance drivers that are partially skill, partially noise.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Turnover differential / game | Net turnovers - high variance but some signal | Game | 🟡 Med | Low | ✅ Done |
| INT rate (off) | Interceptions thrown per attempt | Game | 🟡 Med | Low | ✅ Done |
| Fumble rate (off) | Fumbles per touch | Game | 🟢 Low | Low | ❌ |
| INT rate (def) | Interceptions forced per opponent attempt | Game | 🟡 Med | Low | ❌ |
| Penalty rate | Penalties per game | Game | 🟢 Low | Low | ✅ Done |
| Penalty yards / game | Yardage impact of penalties | Game | 🟢 Low | Low | ❌ |
| False start rate | Offensive discipline proxy | Game | 🟢 Low | Low | ❌ |
| Turnover luck estimate | Compare actual TO diff to expected (fumble recovery regresses to ~50%) | Game | 🟡 Med | Med | ❌ |
| 🆕 Officiating crew tendency | Referee crew's historical flags per game and home/away flag split. `referee` is in nflverse schedules; `load_officials()` has full crews. Crews differ in flag rates, but evidence that this moves totals is thin: cheap to test | Game, Props | 🟢 Low | Low | ❌ |

---

## Domain 4: Quarterback Quality

*The single most important position. Deserves its own feature domain.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| QB Elo / QB-specific rating | Separate Elo track for the starting QB. Glicko-2 or OpenSkill (Part II.3) are alternatives that add an uncertainty estimate; see Domain 14 | Game | 🔴 High | High | ❌ |
| QB EPA/play (career) | Baseline quality signal (`qb_epa` and passer IDs are in PBP) | Game, Props | 🔴 High | Low | ❌ |
| QB EPA/play (rolling L4–L6) | Recent form | Game, Props | 🔴 High | Low | ❌ |
| Passer rating (rolling) | Traditional metric, still informative | Props | 🟡 Med | Low | ❌ |
| Completion % (rolling) | Raw and rolling | Props | 🟡 Med | Low | ❌ |
| CPOE (rolling) | Accuracy over expectation (same PBP column as Domain 1 CPOE) | Game, Props | 🔴 High | Low | ❌ |
| Sack rate taken | How often the QB gets sacked | Props | 🟡 Med | Low | ❌ |
| Scramble rate | How often the QB runs when play breaks down (`qb_scramble` flag in PBP) | Props | 🟡 Med | Low | ❌ |
| Designed rush rate | Designed QB runs per game - key for rushing props (QB carries where `qb_scramble` = 0) | Props | 🟡 Med | Low | ❌ |
| QB rush yards / game (rolling) | Direct input for QB rush prop models | Props | 🔴 High | Low | ❌ |
| QB pass yards / game (rolling) | Direct input for QB pass prop models | Props | 🔴 High | Low | ❌ |
| QB change flag | Is a different QB starting than last week? Historical starters are in nflverse schedules (`home_qb_id` / `away_qb_id`); live use needs the projected starter from depth charts, injury reports, or manual input | Game, What-If | 🔴 High | Med | ❌ |
| QB experience (games started) | Rookie vs veteran signal | Game | 🟢 Low | Low | ❌ |
| 🆕 Time to throw | Average seconds from snap to release (NGS weekly passing stats, 2016+). Helps separate QB-driven sacks and pressure from line-driven ones | Game, Props | 🟡 Med | Low | ❌ |
| 🆕 Pressure-to-sack rate | Sacks divided by pressures faced. Sack avoidance behaves more like a QB trait than a line trait, so it travels with the QB | Game, Props | 🟡 Med | Med | ❌ |
| 🆕 Interception-worthy throw rate | FTN charting's `is_interception_worthy` flag (2022+). A steadier read on turnover risk than actual INTs, which are noisy | Game, Props | 🟡 Med | Med | ❌ |

---

## Domain 5: Schedule & Situational Context

*When, where, and under what circumstances is the game played?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Home field advantage | Binary + strength. If the franchise-level coefficients are fixed across seasons, re-estimate them per season or on a rolling window: league-wide HFA has shrunk over the past decade | Game | 🔴 High | Low | ✅ Done |
| Days rest | Days since last game | Game, Props | 🔴 High | Low | ✅ Done |
| Short week flag | < 7 days rest | Game | 🟡 Med | Low | ✅ Done |
| Post-bye flag | Coming off bye week | Game | 🟡 Med | Low | ✅ Done |
| Travel distance (km) | How far the away team traveled | Game | 🟡 Med | Low | ✅ Done |
| Timezone shift | Crossing time zones | Game | 🟡 Med | Low | ✅ Done |
| Divisional game flag | Division rivalries play differently | Game | 🟡 Med | Low | ✅ Done |
| Primetime flag | Thursday/Sunday/Monday night | Game | 🟢 Low | Low | ✅ Done |
| Dome/outdoor flag | Stadium type | Game, Props | 🟡 Med | Low | ✅ Done |
| Neutral site flag | London, Mexico, etc. | Game | 🟡 Med | Low | ✅ Done |
| Altitude | High-altitude venue (Denver) | Game | 🟢 Low | Low | ✅ Done |
| Season week number | Early vs late season dynamics. Built as `WEEK_NUM`, present in `expanded_152` but not the Logistic `combined_111` set (2026-09-25) | Game | 🟢 Low | Low | ⚠️ Partial |
| Playoff/elimination context | Must-win games may play differently | Game | 🟢 Low | Med | ❌ |
| Rest differential | Team A days rest minus Team B days rest | Game | 🟡 Med | Low | ✅ Done |
| Opponent rest | The other team's rest situation | Game | 🟡 Med | Low | ✅ Done |
| Back-to-back road games | Fatigue / travel compounding | Game | 🟢 Low | Low | ❌ |
| 🆕 Seeding locked / rest-starters risk | Team has clinched, or can't change, its seed late in the season, so starters often sit (especially Week 18). The mirror image of Playoff/elimination context. Few games a year, but large misses when ignored | Game, Props | 🔴 High | Med | ❌ |
| 🆕 Body-clock kickoff time | Kickoff time converted to each team's home time zone (a 1 pm ET kickoff is 10 am for a West Coast team). Extends Timezone shift; circadian research found West Coast teams did better in night games | Game | 🟢 Low | Low | ❌ |
| 🆕 Field surface | Grass vs artificial turf (`surface` in nflverse schedules; already tracked in the stadium reference). More relevant to speed and injury risk than to outcomes | Game, Props | 🟢 Low | Low | ❌ |
| 🆕 League scoring environment & rule era | Season-level league average points/EPA, plus era flags for structural changes (2024 dynamic kickoff; 2025 touchback to the 35 and regular-season overtime change). Keeps totals models from pooling incompatible seasons | Game, Props | 🟡 Med | Low | ❌ |

---

## Domain 6: Weather & Environment

*Physical conditions that affect play style and outcomes.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Temperature (F) | Cold weather affects passing, grip | Game, Props | 🟡 Med | Low | ✅ Done |
| Wind speed (mph) | Affects kicking, deep passing | Game, Props | 🟡 Med | Low | ✅ Done |
| Precipitation flag | Rain/snow binary | Game, Props | 🟡 Med | Low | ✅ Done |
| Weather → feature wiring | OWM weather data wired into prediction features (completed as Priority v1 #1) | Game, Props | 🟡 Med | Low | ✅ Done |
| Wind speed bins | Calm (0–10), moderate (10–20), high (20+) | Game, Props | 🟡 Med | Low | ❌ |
| Cold weather flag | Below 32°F threshold | Props | 🟡 Med | Low | ❌ |
| Indoor override | If dome, weather features zeroed out. Mostly built via `IS_DOME`/`COVERED_STADIUMS` (2026-09-25); the remaining refinement is retractable roofs when closed (`roof` = closed in nflverse schedules), which is null for a meaningful share of future weeks | Game, Props | 🟡 Med | Low | ⚠️ Partial |
| Precipitation type | Rain vs snow (different effects) | Game | 🟢 Low | Med | ❌ |
| Historical weather impact | Team's performance split by weather bucket | Game | 🟢 Low | Med | ❌ |

> **Timing:** training rows use observed game-time weather (OWM backfill), but live predictions will use forecasts. Either train on archived forecasts or measure the forecast-vs-actual gap before trusting weather effects on totals (Part II.7).

---

## Domain 7: Market-Derived Features

*What does the betting market itself tell us?*

> **Philosophical note:** Market features are extremely powerful but create a tension. Using the closing line as a feature means your model is partially *following* the market rather than *disagreeing* with it. This is fine for total projection accuracy but can mask whether your non-market features have genuine signal. **Recommendation:** Train both a "market-aware" and "market-blind" model variant and compare.

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Consensus closing spread | The market's best estimate of team strength. The spread is already ingested and used by the Props model (game_context.py) and is in nflverse schedules (`spread_line`); it just isn't a Game-model feature yet | Game | 🔴 High | Low | ❌ |
| Consensus closing total | Market estimate of combined scoring. Same situation as the spread (`total_line`) | Game, Props | 🔴 High | Low | ❌ |
| Opening line | Where the line opened - often reflects sharp money. Buildable going forward if the odds ledger stores timestamped snapshots | Game | 🟡 Med | Med | ❌ |
| Line movement (open → current) | Direction and magnitude of movement (same ledger dependency as Opening line) | Game | 🟡 Med | Med | ❌ |
| Implied team total | (Total ± Spread) / 2 - crucial for prop context | Props | 🔴 High | Low | ✅ Done |
| Market win probability (no-vig) | De-vigged implied probability from market (moneylines are in nflverse schedules) | Game | 🔴 High | Low | ❌ |
| Reverse line movement flag | Line moves opposite to public money | Game | 🟡 Med | High | ❌ |
| Sharp book (Pinnacle) line | Pinnacle as a separate "sharp" feature | Game | 🟡 Med | Med | ❌ |
| Historical closing line (as prior) | Use last season's closing lines as team-quality prior early in season | Game | 🟡 Med | Med | ❌ |
| 🆕 Player prop line (market) | The prop's own line and odds as a feature: the props version of the market-aware vs market-blind split above. Historical prop lines usually require a paid odds feed | Props | 🔴 High | High | ❌ |
| 🆕 Preseason win total (market) | Early-season team prior that already reflects offseason moves (QB changes, coaching, free agency), unlike last season's closing lines | Game | 🟡 Med | Med | ❌ |

> **Timing:** a closing line exists only at kickoff. If bets are placed earlier, train and run the market-aware model on the line available at bet time, or the backtest will overstate its edge (Part II.7).

---

## Domain 8: Player-Level Features (for Prop Models)

*Individual player performance, usage, and context.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Rolling stat mean (L3, L6, L12) | Per-stat rolling averages at multiple windows | Props | 🔴 High | Med | ✅ Done (L3 + L6) |
| Rolling stat median | More robust to outliers than mean | Props | 🟡 Med | Med | ❌ |
| Rolling stat std dev | Player's own variance - feeds uncertainty bands | Props | 🔴 High | Med | ✅ Done |
| Season average | Full-season baseline | Props | 🟡 Med | Low | ❌ |
| Snap % (rolling) | Playing time trend. Unblocked: nflreadpy ships `load_snap_counts()` (PFR, 2012+). It's keyed by PFR player ID, so it needs an ID crosswalk (`load_ff_playerids()`) to join GSIS-keyed stats, which may be what stalled it | Props | 🔴 High | Med | ❌ |
| Target share (WR/TE) | % of team targets | Props | 🔴 High | Med | ✅ Done |
| Carry share (RB) | % of team carries | Props | 🔴 High | Med | ✅ Done |
| Route participation rate | % of pass plays where WR runs a route. nflverse participation records only the primary receiver's route, so this still needs paid charting | Props | 🟡 Med | High | ❌ |
| Red zone target/carry share | High-value touch distribution | Props | 🟡 Med | Med | ❌ |
| Air yards share | % of team air yards (WR) | Props | 🟡 Med | Med | ✅ Done |
| Yards per route run | Efficiency per opportunity (WR/TE) | Props | 🔴 High | High | ❌ |
| Yards per carry (rolling) | RB efficiency | Props | 🟡 Med | Low | ❌ |
| Yards per target (rolling) | WR/TE efficiency | Props | 🟡 Med | Low | ❌ |
| Matchup: opponent rank vs position | Opponent's defensive rank against this stat | Props | 🔴 High | Med | ✅ Done |
| Matchup: opponent EPA allowed vs position | More granular matchup quality | Props | 🔴 High | Med | ✅ Done |
| Home/away split | Player's home vs away performance | Props | 🟡 Med | Low | ❌ |
| Indoor/outdoor split | Dome vs open-air | Props | 🟡 Med | Low | ❌ |
| vs. winning teams split | Performance against good teams | Props | 🟢 Low | Low | ❌ |
| Game script proxy (spread) | Implied game flow from spread | Props | 🔴 High | Low | ✅ Done |
| Implied team total | (Total ± Spread) / 2 - volume expectation | Props | 🔴 High | Low | ✅ Done |
| Weather × stat interaction | Wind + cold suppress passing, boost rushing | Props | 🟡 Med | Med | ❌ |
| Return from injury flag | First game back - usage often limited | Props, What-If | 🟡 Med | Med | ❌ |
| Weeks since injury | Ramp-up trajectory | Props, What-If | 🟡 Med | Med | ❌ |
| ↺ Touch share (RB) | % of team touches (carries + targets). Built per the 2026-06-10 changelog entry; had no row | Props | 🔴 High | Med | ✅ Done |
| 🆕 Expected yards / TDs / fantasy points | nflverse `load_ff_opportunity()`: what an average player would produce from the same touches and targets. Separates opportunity from efficiency | Props | 🔴 High | Low | ❌ |
| 🆕 Expected vs actual TDs | TDs are noisy and regress toward opportunity-based expectation: the touchdown version of the turnover luck estimate. Directly relevant to anytime-TD props | Props | 🟡 Med | Low | ❌ |
| 🆕 Age / aging curve | Position-specific age adjustment (running backs decline earliest). Matters most for season-long priors | Props | 🟢 Low | Low | ❌ |
| 🆕 Draft capital (rookie prior) | Draft round and pick as a prior for rookies with thin samples; it strongly predicts early opportunity | Props | 🟡 Med | Low | ❌ |
| 🆕 Coverage matchup (man/zone) | Receiver performance vs man vs zone, against the opponent's man/zone rate. Participation has `defense_man_zone_type` historically; in-season coverage needs a paid charting source | Props | ❓ Unknown | High | ❌ |

---

## Domain 9: Roster & Personnel - The What-If Domain

*Who is playing, who isn't, and what does it mean?*

This is the domain that powers the **Scenario Engine** (ROADMAP W4.5). It's the most complex and the most differentiating capability.

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Player WAR (wins above replacement) | How many wins does this player add? nflWAR is the reproducible template (Part II.2) | What-If, Game | 🔴 High | High | ❌ |
| On/off EPA split | Team EPA with vs without this player. Needs who-was-on-the-field data: nflverse participation (2016+, released after each season). RAPM is the regularized version (Part II.2) | What-If, Game | 🔴 High | High | ❌ |
| Positional importance weight | QB > Edge > WR > RB for team impact | What-If, Game | 🟡 Med | Med | ❌ |
| Injury status (Out/Doubtful/Q/Probable) | Current game status via `load_injuries()`. Its source switched to ESPN in 2025 and the structure changed; "Probable" was retired in 2016, so categories shift across eras | What-If, Game, Props | 🔴 High | Med | ❌ |
| Estimated play probability | Probability player actually plays given status | What-If | 🟡 Med | Med | ❌ |
| Backup quality rating | How good is the replacement? Replacement level is the hard part (Part II.2); OpenSkill μ/σ (Domain 14) is one route | What-If, Game | 🔴 High | High | ❌ |
| Usage redistribution model | If RB1 out → RB2 gets X% of carries, RB3 gets Y% | What-If, Props | 🔴 High | High | ❌ |
| Target tree redistribution | If WR1 out → WR2/TE1/RB target shares shift | What-If, Props | 🔴 High | High | ❌ |
| Cumulative injury impact | Sum of WAR-weighted absences on a roster | What-If, Game | 🔴 High | High | ❌ |
| O-line health index | Composite of OL starters available | Game, Props | 🟡 Med | High | ❌ |
| Historical with/without record | Team's ATS record with and without this player | What-If | 🟡 Med | Med | ❌ |
| Depth chart stability | How much roster churn has occurred recently | Game | 🟢 Low | Med | ❌ |
| Games together (unit cohesion) | How many games has the current OL/WR corps played together | Game | 🟢 Low | High | ❌ |
| 🆕 Practice participation trajectory | DNP → Limited → Full pattern through the week. nflverse carries practice status; the full day-by-day pattern may need the daily reports | Props, What-If | 🔴 High | Med | ❌ |

### How the What-If Scenario Engine Would Work (illustrative numbers)

```
Scenario Input:
  "CMC is OUT for SF @ BAL"

Step 1 - Player Impact Quantification:
  CMC WAR = 1.8 wins
  CMC on/off EPA split = +0.09 EPA/play

Step 2 - Team Adjustment:
  SF offensive rating: 82.3 → 76.1 (adjusted)
  SF win probability: 29% → 22%
  SF implied team total: 21.5 → 18.8

Step 3 - Usage Redistribution:
  CMC carries/game: 18.4 → 0
  Jordan Mason carries/game: 8.2 → 19.6
  Deebo Samuel touches: +2.4

Step 4 - Prop Re-Forecast:
  Mason rush yards projection: 62.1 → 88.4
  Purdy pass attempts projection: 32.1 → 35.8
  Purdy pass yards projection: 242 → 258

Step 5 - Edge Re-Calculation:
  BAL -4.5 edge: +3.1% EV → +5.8% EV
  Mason rush OVER 62.5: no edge → +4.2% EV (new line)
```

**Method hooks (Part II):**

- **Step 1 (player impact):** nflWAR and RAPM-style on/off (II.2), with OpenSkill μ/σ as a cross-check (Domain 14).
- **Step 3 (usage redistribution):** snap counts plus historical with/without usage shares.
- **Step 4 (prop re-forecast):** count distributions and whole-game simulation, so correlated props stay consistent with each other (II.5).
- **Step 5 (edge):** only as good as the calibration underneath it. Track closing line value on every edge taken (II.6).

---

## Domain 10: Coaching & Scheme

*How does the coaching staff affect game outcomes and player usage?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Head coach win % (career) | Baseline coaching quality (head coaches are in nflverse schedules) | Game | 🟢 Low | Low | ❌ |
| HC tenure (years with team) | Stability / familiarity | Game | 🟢 Low | Low | ❌ |
| Offensive coordinator tenure | New OC = scheme adjustment period | Game, Props | 🟡 Med | Med | ❌ |
| Play-calling tendency (pass/run) | Coaching scheme signature. Overlaps the Domain 1 pass-rate rows; build once, reuse | Props | 🟡 Med | Med | ❌ |
| Aggressiveness (4th down go rate) | Risk tolerance proxy | Game | 🟢 Low | Med | ❌ |
| Historical ATS performance | Some coaches consistently beat/miss spreads | Game | 🟢 Low | Med | ❌ |
| Coaching matchup history | H2H coaching record | Game | 🟢 Low | Med | ❌ |
| New coaching staff flag | First-year HC/OC/DC (HC from schedules; OC/DC need another source) | Game | 🟡 Med | Low | ❌ |

---

## Domain 11: Historical Trends & Situational Patterns

*Meta-features about when/how teams perform differently.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Win streak / loss streak | Momentum proxy. Research finds no between-game momentum once team quality is accounted for, so this is an ablation candidate (Part II.8) | Game | 🟢 Low | Low | ✅ Done |
| Win % (season) | Current season record | Game | 🟢 Low | Low | ✅ Done |
| ATS record (season) | Betting-specific performance (spreads and results are in nflverse schedules) | Game | 🟡 Med | Low | ❌ |
| Over/Under record (season) | Tendency to go over or under (same source) | Game | 🟡 Med | Low | ❌ |
| Performance after loss | Bounce-back tendency | Game | 🟢 Low | Low | ❌ |
| Performance as favorite/underdog | Does the team cover more as dog or fav? | Game | 🟡 Med | Med | ❌ |
| Performance by spread bucket | ATS record in -3 to -7 range vs -7+ etc. | Game | 🟢 Low | Med | ❌ |
| Scoring by quarter | Is the team a fast or slow starter? | Game | 🟢 Low | Med | ❌ |
| Season week performance | Early vs mid vs late season form | Game | 🟢 Low | Low | ❌ |
| ↺ Average score differential | `avg_score_diff`. Built per the 2026-06-04 changelog entry; had no row | Game | 🟡 Med | Low | ✅ Done |
| ↺ Close game % | `close_game_pct`. Built per the 2026-06-04 changelog entry; had no row | Game | 🟢 Low | Low | ✅ Done |
| 🆕 Pythagorean win % / luck gap | Expected win % from points scored and allowed vs actual win %. Teams winning more than their scoring supports tend to fall back | Game | 🟡 Med | Low | ❌ |
| 🆕 One-score game record | Record in games decided by 8 points or fewer; regresses hard toward .500 | Game | 🟡 Med | Low | ❌ |

---

## Domain 12: Special Teams & Field Position

*The third phase of the game, missing from earlier revisions entirely.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| 🆕 Special teams EPA/play | Combined kicking, punting, and return EPA from PBP | Game | 🟡 Med | Low | ❌ |
| 🆕 Field goal accuracy over expected | Kicker make rate vs expectation for distance and conditions (wind, altitude) | Game, Props | 🟡 Med | Med | ❌ |
| 🆕 Punt & return efficiency | Net punting and return EPA: the field-position flippers | Game | 🟢 Low | Low | ❌ |
| 🆕 Average drive start (off/def) | Where drives begin. Blends special teams, turnovers, and defense into one field-position number | Game | 🟡 Med | Low | ❌ |
| 🆕 Kicker opportunity | FG and extra-point attempts per game, driven by red-zone stalls (Red zone TD % already exists). High signal for kicker props specifically | Props | 🔴 High | Low | ❌ |

> **Rule era:** the 2024 dynamic kickoff moved average post-kickoff starting position from about the 26 to the 30, and 2025 moved touchbacks to the 35. Kickoff and field-position features from earlier seasons don't transfer directly; use the rule-era flag (Domain 5).

---

## Domain 13: Unit Matchups & Interactions

*Features built from both teams at once. Tree models can find some of these on their own; linear models need them spelled out.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| 🆕 Pass rush vs pass protection | Defense's pressure rate generated × offense's pressure rate allowed (depends on Pressure rate, Domain 2) | Game, Props | 🟡 Med | Med | ❌ |
| 🆕 Unit efficiency differentials | Offense pass EPA vs opposing defense pass EPA allowed (same for rush), ideally on opponent-adjusted EPA | Game | 🟡 Med | Low | ❌ |
| 🆕 Projected total plays | Both teams' pace combined, adjusted for expected game script from the spread. The volume driver behind totals and every counting-stat prop | Game, Props | 🔴 High | Low | ❌ |

---

## Domain 14: Rating-System Outputs

*Outputs of the rating engines in Part II.3, used as model inputs.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| ↺ Team Elo rating | Existing Elo engine (v1–v3): baseline team strength and the input to strength of schedule | Game | 🔴 High | Low | ✅ Done |
| ↺ Strength of schedule (opponent Elo) | Average opponent Elo. In the feature set but had no row | Game | 🟡 Med | Low | ✅ Done |
| 🆕 Team Glicko-2 rating (+ RD, volatility) | Elo plus an uncertainty estimate and a volatility term. Natural fit for the team layer, since games are always two sides | Game | ❓ Unknown | Med | ❌ |
| 🆕 Keener / Colley / Massey ratings | Linear-algebra rankings with no probability model underneath. Useful as ensemble inputs because their assumptions differ from everything else | Game | 🟢 Low | Low | ❌ |
| 🆕 Player skill rating (OpenSkill μ/σ) | Per-player rating fed by per-play performance scores (EPA, CPOE, RYOE, pressure), updated free-for-all within position group. Unproven in football (Part II.3) | What-If, Props, Game | ❓ Unknown | High | ❌ |
| 🆕 Rating uncertainty as a feature | σ or RD passed to the model directly, so it can learn to trust ratings less early in the season or after a QB change | Game, Props | ❓ Unknown | Low | ❌ |

---

# Part II — Methods & Modeling Reference

*How features get measured, combined, and checked. Section numbers are referenced from Part I.*

## II.1 Team strength: measurement methods

| Method | What it does | Role in Gridiron Edge |
|--------|-------------|----------------------|
| Elo | One strength number per team, nudged after every game | In place (v1–v3) |
| SRS (Simple Rating System) | Average point margin adjusted for schedule, solved as one linear system | Cheap, transparent baseline |
| Massey | Least-squares ratings fit to game margins | Ensemble input (Domain 14) |
| Colley matrix | Win/loss only (no margin), pulled toward .500 and adjusted for schedule; ratings always average 0.500 | Ensemble input (Domain 14) |
| Keener | Strength is the dominant eigenvector of a score-based "who dominated whom" matrix (the same math as PageRank) | Ensemble input (Domain 14) |
| Opponent-adjusted EPA | Regression on play-level EPA with an offense term and a defense term, shrunk toward average when samples are small | Core upgrade to Domains 1–2 |
| DVOA (FTN) | Opponent- and situation-adjusted success; paywalled | Optional benchmark |
| Logistic regression / Bayesian hierarchical | Win probability; hierarchical versions shrink team strength toward the mean on small samples | In place (logistic v1–v4) |
| Gradient-boosted trees + SHAP | Learn nonlinear interactions; SHAP shows which features drive each prediction | In place (XGBoost, Random Forest) |
| Monte Carlo simulation | Distributions instead of single numbers | In place for seasons and playoffs; extend to games and props (II.5) |

## II.2 Player value and attribution

| Method | Isolates the player? | Reproducible? | Best fit |
|--------|---------------------|---------------|----------|
| Raw rate stats (YPA, YPC, TD rate) | No | Yes | Baselines only |
| EPA decomposition | Partly (cleanest for QBs) | Yes | Quarterbacks |
| Expectation metrics (CPOE, RYOE, expected YAC, separation) | Yes, against an average player in the same situation | Yes (NGS aggregates are public) | Skill positions |
| Grading (PFF) | Yes, via human judgment | No | OL and DBs, where box scores are empty; a black box |
| Plus-minus / RAPM (and HBPM) | Yes, net of teammates and opponents | Yes, given participation data | Offensive line, defense |
| nflWAR | Yes, in wins | Yes | Offensive player value (Domain 9 WAR) |
| STRAIN | Yes, for pass rushers | Yes, but needs raw tracking data | Pass rush |

- **EPA is the atomic input; a "score" is an aggregation choice.** The real decision is formulaic aggregation (reproducible, debuggable) vs discretionary grading (richer per-play judgment, but it can't be rebuilt or audited).
- **Per-game aggregation is itself a modeling decision.** Simple averages, snap-weighted, and leverage-weighted (discounting garbage time) produce different rankings from the same plays.
- **Replacement level is the hard part** of WAR and of Domain 9's backup quality. Define it explicitly, for example as the average production of players ranked outside the top N at the position.
- **Frontier:** raw tracking data plus graph neural networks learn player interactions directly. Public tracking data is limited to released Big Data Bowl windows, so treat this as research, not production.

## II.3 Rating systems

| System | Update style | Multi-player teams | Uncertainty | Availability | Role in Gridiron Edge |
|--------|-------------|--------------------|-------------|--------------|----------------------|
| Elo | Online, game by game | No | None | Open | In place |
| Glicko-2 | Online, by rating period | No (1v1) | Rating deviation (RD) + volatility | Open | Team layer: NFL games are always two sides |
| OpenSkill (Weng-Lin) | Online | Yes | μ (skill) and σ (uncertainty); τ keeps ratings from freezing | Open source | Player layer |
| TrueSkill | Online | Yes | μ, σ | Name and algorithm restricted to non-commercial or Xbox use | Avoid |
| TrueSkill 2 | Online | Yes | Adds individual performance signals | Paper only (proprietary) | Design pattern to copy |
| WHR (Whole History Rating) | Batch: re-estimates the whole history at once | 1v1 as published | Yes | Open (paper and implementations) | Offline benchmark for where live ratings should converge |
| Kalman filter / state-space | Online | Custom | Yes | General technique | Escape hatch: custom season decay, multi-part skill (e.g., separate pass and run), covariates |
| Keener / Colley / Massey | Batch | No | None | Open | Ensemble diversity |

- **Common root.** Elo, Glicko, TrueSkill, WHR, and OpenSkill's Bradley-Terry option all descend from the Bradley-Terry paired-comparison model (1952). Keener, Colley, and Massey are linear algebra with no probability model underneath, which is exactly why they add diversity to an ensemble.
- **Proposed player-rating pattern (unproven in football):**
  1. Score every player on every play with a position-appropriate metric. QB: EPA and CPOE. RB: RYOE. WR/TE: EPA per target or expected-vs-actual yards. Pass rusher: pressure.
  2. Run OpenSkill in free-for-all mode *within position groups*, using those scores as the results.
  3. Feed μ and σ to the models (Domain 14). Use WHR offline to check where the live ratings should be heading.
- **Precedent:** PandaSkill (esports) uses this structure, with a per-player performance score feeding OpenSkill in free-for-all mode. Football-specific precedent is thin.
- **Pitfall:** putting all 11 offensive players on one "team" against the defense mostly rebuilds a diluted team rating. Everyone moves together unless something else separates them.
- **Sequencing:** test Glicko-2 at the team layer first (cheap and directly comparable to the existing Elo) before investing in player-level OpenSkill.

## II.4 Method → catalog map

| Catalog feature (Part I) | Method | Data |
|--------------------------|--------|------|
| Player WAR (D9) | nflWAR | PBP |
| On/off EPA split (D9) | RAPM / HBPM | Participation |
| Backup quality rating (D9) | Explicit replacement level + OpenSkill μ/σ | PBP, participation |
| Pressure rate (D2) | Charted `was_pressure`; STRAIN only if raw tracking ever becomes available | Participation (historical), PFR (in-season) |
| QB Elo / QB-specific rating (D4) | Glicko-2 or OpenSkill track per QB | PBP, schedules |
| Def DVOA (D2) | DIY opponent-adjusted EPA | PBP |
| Turnover luck estimate (D3) | Actual vs expected fumble recoveries (about 50%) | PBP |
| Usage and target tree redistribution (D9) | Historical with/without usage shares | Snap counts, player stats |
| Player skill rating (D14) | OpenSkill free-for-all (II.3) | PBP + derived scores |

## II.5 Combining, calibrating, validating

- **Already in place:** Brier score, log loss, ECE, ROC-AUC, calibration curves, and champion/challenger selection.
- **Walk-forward validation.** Random or stratified k-fold on game data lets future games train the model that "predicts" past ones. If StratifiedKFold drives model selection or tuning anywhere, switch to chronological splits: train through week N, test on week N+1, roll forward (or use season-level folds). This is the most common source of backtests that look better than live results.
- **Recalibration.** When ECE or the reliability curve shows systematic drift, fit isotonic regression or Platt scaling on out-of-time data.
- **Ensembling / stacking.** Weighted averages (weights from out-of-time log loss), a simple regularized logistic meta-model trained on base-model outputs, or Bayesian model averaging.
- **Market-aware vs market-blind variants** (Domain 7 note). Keep both; the blind model is how you learn whether your features carry anything the market doesn't already know.
- **Count distributions for props.** Receptions and TDs are counts (Poisson or negative binomial, zero-inflated where needed); yardage is skewed and continuous. The market prices a distribution, not a mean.
- **Correlated outcomes.** A QB's passing yards and his receivers' yards move together. Pricing props one at a time misprices anything combined, such as same-game parlays. Simulating whole games, by extending the existing Monte Carlo engine, keeps joint outcomes consistent.
- **Conformal prediction.** Wraps any model to produce intervals with guaranteed coverage and no distribution assumptions: a check on the parametric uncertainty coming from ratings and count models.

## II.6 Benchmarks and red flags

- Closing spreads picked straight-up winners about 66% of the time over 2007–2020 (opening lines about 63.5%), and nfelo runs at about the same level. Matching the close is the realistic accuracy ceiling.
- Claims of 85–90%+ accuracy almost always mean target leakage: the game's own stats used to "predict" it.
- For betting, the per-bet yardstick is **closing line value (CLV)**: did the price you took beat the final pre-game price? Consistently positive CLV is stronger evidence of an edge than a short-run win rate.
- Judge probabilities with log loss, Brier, and calibration, not hit rate.
- **Sample size.** At about 285 games a season, even ten seasons is under 3,000 games: small for the 152 correlated features in the current `expanded_152` set. The 2026-06-04 result (challengers rejected) and the flat Brier scores across logistic variants point the same way. New *kinds* of information will help more than more variants of the same rolling team stats.

## II.7 Feature construction rules

1. **As-of timing.** Build every feature only from what's known when the prediction is made: the line at bet time (the close exists only at kickoff), the injury designation from the report actually used, the projected starter rather than the actual starter, and forecast weather rather than observed weather. Where that isn't possible, measure the gap explicitly.
2. **Garbage time.** Drop or down-weight plays in decided games (for example, win probability outside 10–90%), plus kneels and spikes, before building EPA features.
3. **Opponent adjustment.** Adjust efficiency for the opponents faced (II.1) instead of leaving it to a single strength-of-schedule column.
4. **Early-season priors.** Rolling windows are thin in September. Blend in prior-season values (regressed toward the mean) or market priors (preseason win totals), shifting weight to current-season data as games accumulate.
5. **Recency.** Test exponential decay against fixed L3/L6 windows.
6. **Stability check.** Before trusting a feature, check how well it predicts itself later (week to week, or split-half). Fumble recoveries barely predict themselves; per-play efficiency does much better. Unstable features need heavier shrinkage toward the mean.
7. **Source consistency.** When the training source differs from the live source (participation-based pressure historically vs PFR in-season), calibrate one to the other before mixing. Otherwise the model learns one scale and gets fed another.
8. **Rule and category breaks.** Kickoffs changed in 2024 (dynamic kickoff; average post-kickoff start moved from about the 26 to the 30) and again in 2025 (touchback to the 35). Regular-season overtime changed in 2025. "Probable" disappeared from injury reports in 2016. Flag eras rather than pooling blindly.

## II.8 Factors that don't hold up well

| Factor | Why to be skeptical | Where it lives |
|--------|--------------------|----------------|
| Win/loss streaks ("momentum") | Research finds no between-game momentum once team quality is accounted for | D11, built: ablation candidate |
| Raw turnover margin as a predictor | Fumble recoveries are close to coin flips; use luck-adjusted versions | D3, built: pair with turnover luck |
| Time of possession | Mostly a result of leading (teams ahead run the clock), not a cause | D1, missing: low priority |
| Passer rating | Tracks winning poorly next to EPA and CPOE | D4, missing |
| Coaching ATS history, revenge games, look-ahead/letdown spots | Popular in handicapping, little reliable evidence | D10–D11: test before building |
| Points allowed per game | Noisy next to per-play measures | D2, missing |
| Anything reporting 85–90%+ accuracy | The signature of leakage | — |

## II.9 Data sources

nflreadpy (the successor to nfl_data_py) follows nflreadr's `load_*` naming.

| Source | What it adds | Coverage and timing | Notes |
|--------|-------------|--------------------|-------|
| `load_pbp()` | EPA, WP, CPOE, `xpass`/`pass_oe`, air yards, YAC, scramble flags, special teams | 1999+, updated in-season | Core |
| `load_schedules()` | Results, spread/total/moneylines, rest, roof, surface, starting QBs, head coaches, referee | Updated in-season | Starting QB is the actual starter; use projected starters live |
| `load_snap_counts()` | Offense, defense, and special teams snap share | 2012+ (PFR) | Keyed by PFR player ID; needs the `load_ff_playerids()` crosswalk |
| `load_participation()` | Players on the field per play, personnel, box count; recent seasons add `was_pressure`, `time_to_throw`, man/zone, coverage shell, primary receiver's route | 2016+; published after each season, not in-season | On/off, RAPM, pressure, coverage. Check field coverage per season |
| `load_nextgen_stats()` | Weekly passing, rushing, and receiving aggregates (time to throw, CPOE, RYOE, separation) | 2016+ | |
| `load_pfr_advstats()` | Pressures, blitzes, drops, broken tackles (pass, rush, rec, def) | 2018+ | Likely in-season pressure source; confirm update timing |
| `load_ftn_charting()` | Play action, motion, blitzers, interception-worthy throws, catchable balls, drops | 2022+, typically within about 48 hours of games | |
| `load_injuries()` | Game status and practice participation | Source switched to ESPN in 2025; structure changed | Verify columns and coverage first |
| `load_ff_opportunity()` | Expected yards, TDs, and fantasy points | Check coverage | Props |
| `load_depth_charts()`, `load_draft_picks()`, `load_contracts()`, `load_officials()` | Depth order, draft capital, contracts, officiating crews | Depth charts 2001+, drafts 1980+ | |
| FTN (DVOA) | Opponent-adjusted efficiency | Paywalled | Optional benchmark |
| PFF | Player grades | Paid | Not reproducible |
| NGS raw tracking | Player positions 10 times per second | Released Big Data Bowl windows only | STRAIN, GNN research |
| OpenWeatherMap | Weather | Integrated | Observed (backfill) vs forecast (live); see II.7 |
| DraftKings odds ledger | Lines and odds | Integrated, going forward | If snapshots are timestamped, enables line-at-bet-time features |
| Paid odds feed | Historical prop lines | Paid | Needed for prop market features |

---

# Part III — Priorities & Status

## Priority Matrix v1: Top 15 Features to Add Next (original, kept for history)

Ranked by signal × cost ratio. These feed directly into PLAN.md as actionable tasks.

| Priority | Feature | Domain | Model | Why |
|----------|---------|--------|-------|-----|
| 1 | Wire weather into prediction features | Weather | Game | Ingest exists, feature doesn't - pure wiring (DONE) |
| 2 | Wire dome/neutral/altitude into features | Schedule | Game | Already in schema, just needs end-to-end (DONE) |
| 3 | Success rate (pass/rush) | Offense | Game | Low cost, adds dimension beyond EPA (DONE) |
| 4 | 3rd down conversion % (off + def) | Off/Def | Game | Easy from PBP, strong signal (DONE) |
| 5 | Red zone TD % (off + def) | Off/Def | Game | Easy from PBP, affects scoring (DONE) |
| 6 | Turnover differential / game | Turnovers | Game | Simple, some signal (DONE) |
| 7 | Sack rate (off + def) | Off/Def | Game, Props | Easy from PBP, affects QB props (DONE) |
| 8 | Implied team total | Market | Props | Pure math once you have spread + total ✅ Done (game_context.py) |
| 9 | Rolling stat mean (L6) per player | Player | Props | Foundation for all prop models ✅ Done (L3 + L6, rolling.py) |
| 10 | Rolling stat std dev per player | Player | Props | Feeds uncertainty bands ✅ Done (rolling.py) |
| 11 | Snap % (rolling) per player | Player | Props | Usage = volume = projections Deferred (nflreadpy doesn't expose snap counts) |
| 12 | Matchup: opponent rank vs position | Player | Props | The #1 prop-specific feature ✅ Done (matchup.py) |
| 13 | QB rush yards/game (rolling) | QB | Props | Direct input for first prop model |
| 14 | Rest differential | Schedule | Game | Already have each team's rest - just subtract (DONE) |
| 15 | Explosive play rate | Offense | Game | Captures big-play ability beyond EPA mean (DONE) |

Items 1–7 and 14–15 are complete (feature engineering done; the expanded set has since grown to `expanded_152`, see Domain summary above).
Items 8–13 are W4 player data (start once player game logs are ingested).

**Status as of 2026-09-24:** 13 of 15 done. #11 is unblocked (`load_snap_counts()` exists; see Domain 8) and #13 is still open. Both carry into v2.

## Priority Matrix v2 (proposed, not yet adopted)

*Ranked by signal × cost, weighted toward new kinds of information over more variants of existing rolling stats (Part II.6).*

**Prerequisite (process, not a feature):** confirm that model selection and tuning use chronological splits (Part II.5). If they don't, every ranking below rests on optimistic backtests.

| Priority | Feature | Domain | Model | Why |
|----------|---------|--------|-------|-----|
| 1 | QB change flag / starting QB identity | QB (D4) | Game, What-If | The biggest blind spot: QB changes are when rolling team stats are most wrong. Historical starters are free in schedules |
| 2 | QB EPA/play (rolling) + CPOE fix | QB (D4), Offense (D1) | Game, Props | Already in PBP; also clears the ⚠️ on CPOE |
| 3 | Snap % (rolling) | Player (D8) | Props | Unblocked by `load_snap_counts()`; needs the ID crosswalk |
| 4 | Garbage-time filter on EPA features | Construction (II.7) | Game, Props | Cleans every existing EPA column at once |
| 5 | Opponent-adjusted EPA (off + def) | Offense/Defense (D1–D2) | Game, Props | Biggest upgrade to features already built; replaces sourcing DVOA |
| 6 | Market spread, total, and no-vig WP as Game features | Market (D7) | Game | Data already in hand; enables the market-aware vs market-blind comparison |
| 7 | QB rush yards/game (rolling) | QB (D4) | Props | v1 #13, still open |
| 8 | Projected total plays | Matchups (D13) | Game, Props | Volume driver for totals and every counting-stat prop |
| 9 | Expected yards/TDs (`load_ff_opportunity()`) | Player (D8) | Props | Free; separates opportunity from efficiency |
| 10 | Injury status + practice participation | Roster (D9) | Props, What-If | Check the ESPN-sourced structure first |
| 11 | Pressure rate | Defense (D2) | Game, Props | Participation historically, PFR in-season; calibrate between them |
| 12 | Luck-regression set: turnover luck, Pythagorean gap, one-score record, expected vs actual TDs | D3, D8, D11 | Game, Props | Cheap; each targets a known regression-to-the-mean effect |
| 13 | Special teams EPA + average drive start | Special Teams (D12) | Game | An entire phase of the game is currently missing |
| 14 | Seeding-locked flag | Schedule (D5) | Game, Props | Rare, but large misses in Week 18 |
| 15 | Team Glicko-2 rating + RD as a feature | Ratings (D14) | Game | Cheapest test of the rating-systems thread, directly comparable to Elo |

**Cleanup alongside:** ablation-test the win/loss streak feature, and close the duplicate "Pace tendency" row (D10) against "Plays per game / pace" (D1).

**Status, 2026-09-25:** the process prerequisite is already met — all model selection and tuning use `TimeSeriesSplit`; there is no `StratifiedKFold`, `KFold`, or `train_test_split` anywhere in `src/`. This matrix is superseded for game-model rows by the "Game-Model Build Queue" below, adopted into ROADMAP.md Tier 5 #16. It remains proposed, not yet adopted, for its prop-only rows (#3, #7, #9, #10).

---

## Game-Model Build Queue (adopted 2026-09-25)

*Every Part I row targeting the Game model (Win: `HOME_WIN` — Elo, Logistic, Random Forest, XGBoost; Total: `ACTUAL_TOTAL` — Random Forest, XGBoost) that is ❌ Missing or ⚠️ Partial, mapped to its ROADMAP Tier 5 #16 phase/unit or to an explicit exclusion. Program rules, phase definitions, and acceptance criteria live in ROADMAP.md Tier 5 #16. Prop-only rows (Model Target excludes Game) are out of scope for this program and stay in Priority Matrix v2 or the general backlog.*

### Phase A — free, data already on disk

**A1 Construction fixes:** CPOE (D1, ⚠️ Partial — NaN-aware mean plus early-season prior); Fumble rate, off (D3); INT rate, def (D3).

**A2 Opponent-adjusted EPA and unit interactions:** Opponent-adjusted EPA/play, pass+rush (D1 🆕); Opponent-adjusted def EPA/play, pass+rush (D2 🆕); Unit efficiency differentials (D13 🆕); Projected total plays, market-blind variant (D13 🆕).

**A3 QB and coach sidecar:** QB EPA/play, career (D4); QB EPA/play, rolling L4–L6 (D4); CPOE, rolling (D4); QB change flag (D4); QB experience (D4); Head coach win %, career (D10); HC tenure (D10); New coaching staff flag (D10).

**A4 Situational context:** Season week number (D5 — **status correction:** already built as `WEEK_NUM`, present in `expanded_152` but not `combined_111`; reclassify ⚠️ Partial, not ❌); Back-to-back road games (D5); Body-clock kickoff time (D5 🆕); Field surface (D5 🆕); League scoring environment & rule era (D5 🆕, as a continuous season-to-date signal — fixed rule-era flags exist only in holdout seasons and can't be learned); Wind speed bins (D6 🆕); Indoor override (D6 — **status correction:** mostly built via `IS_DOME`/`COVERED_STADIUMS`; reclassify ⚠️ Partial; the remaining closed-roof refinement is low priority because `roof` is null for a meaningful share of future weeks); Season week performance (D11); Pythagorean win % / luck gap (D11 🆕); One-score game record (D11 🆕); Special teams EPA/play (D12 🆕, from stored play-by-play once special-teams play types are included in aggregation); Punt & return efficiency (D12 🆕).

**A5 Rating ensemble inputs:** QB Elo / QB-specific rating (D4, via Glicko-2/OpenSkill; needs the same lineage treatment as team Elo); Team Glicko-2 rating + RD + volatility (D14 🆕); Keener / Colley / Massey ratings (D14 🆕); Rating uncertainty as a feature (D14 🆕).

**A6 Market-aware variant** (separate model identity from the market-blind champion; needs the `DECISIONS.md` entries in ROADMAP Tier 5 #16 rule 4): Consensus closing spread (D7); Consensus closing total (D7); Market win probability, no-vig (D7); Historical closing line as a prior (D7); ATS record, season (D11); Over/Under record, season (D11); Performance as favorite/underdog (D11).

**A7 Research only, low evaluation power:** Playoff/elimination context (D5); Seeding locked / rest-starters risk (D5 🆕) — both via `sim/`; affects too few holdout games (~48) to reliably clear the acceptance bar without several seasons of data.

### Phase B — free, needs an nflverse play-by-play re-download (ask before running)

**B1:** Scoring rate (D1, points per drive — needs a drive count, which the current play-by-play keep-list doesn't retain); Penalty yards / game (D3); False start rate (D3); Turnover luck estimate (D3); Aggressiveness, 4th-down go rate (D10); Average drive start, off/def (D12 🆕); Field goal accuracy over expected (D12 🆕, also needs distance/attempt modeling).

**B3:** Pressure rate (D2 — calibrate historical participation `was_pressure` against in-season PFR advanced stats); Time to throw (D4 🆕); Pressure-to-sack rate (D4 🆕); Pass rush vs. pass protection (D13 🆕, depends on Pressure rate).

### Phase C — costs money, last

**C1:** Precipitation type (D6, low priority; revisit alongside live forecast weather).

**C2:** Opening line (D7); Line movement, open→current (D7); Sharp book (Pinnacle) line (D7); Preseason win total, market (D7 🆕, no confirmed free historical source).

**C3 (paid benchmark only, not a model feature):** Def DVOA, if sourced (D2).

### Excluded, with reason

| Feature | Domain | Reason |
|---|---|---|
| Time of possession | D1 | Thin evidence (II.8): mostly an effect of leading, not a cause |
| Points allowed / game | D2 | Thin evidence (II.8): noisy next to per-play measures |
| 🆕 Officiating crew tendency | D3 | `referee` is null for upcoming games before kickoff; fails the live-availability rule |
| 🆕 Interception-worthy throw rate | D4 | FTN charting starts 2022; leaves no training season before the 2023 holdout |
| Reverse line movement flag | D7 | No data source; High cost |
| Offensive coordinator tenure | D10 | No free source for OC/DC identity or tenure |
| Play-calling tendency (pass/run) | D10 | Props-only Model Target; overlaps D1 PROE work, not in scope here |
| Pace tendency (plays/game) | D10 | Duplicate of D1 "Plays per game / pace" (already ✅ Done); close the row, don't rebuild |
| Historical ATS performance | D11 | Thin evidence (II.8-adjacent): popular in handicapping, little reliable evidence |
| Coaching matchup history | D11 | Same as above |
| Performance after loss | D11 | Thin evidence; bounce-back tendency is a similar unverified pattern to momentum (II.8) |
| Performance by spread bucket | D11 | Explicitly flagged "test before building" with no test done yet |
| Scoring by quarter | D11 | Needs a quarter/drive field not in the current play-by-play keep-list; low signal for the cost |
| Cold weather flag | D6 | Model Target is Props only; out of scope for this game-only program |
| Historical weather impact | D6 | Thin evidence; Medium cost for Low signal |
| Player skill rating (OpenSkill μ/σ) | D14 | Unproven in football (Part II.3); Research Backlog, not this program |
| Player WAR; On/off EPA split; Positional importance weight; Injury status; Estimated play probability; Backup quality rating; Usage/target-tree redistribution; Cumulative injury impact; O-line health index; Historical with/without record; Depth chart stability; Games together; 🆕 Practice participation trajectory | D9 | Roster/injury/what-if features move to ROADMAP Tier 6 #18 (injury ingestion) or Tier 7 (scenario engine); this program is game-model features only |

---

## Summary Statistics

*Exact counts of Part I catalog rows as of 2026-09-25 (recounted by script after closing the duplicate D10 "Pace tendency" row and correcting 2 Status values).*

| Metric | Count |
|--------|-------|
| Catalog rows | 179 (178 unique; "Implied team total" is cross-listed in Domains 7 and 8) |
| Domains | 14 (11 original + 3 new) |
| ✅ Done | 57 |
| ⚠️ Partial | 3 (CPOE; Season week number; Indoor override) |
| ❌ Missing | 119 |
| 🔴 High signal | 49 |
| ❓ Unknown signal | 4 |
| Low cost | 115 |
| High signal, Low cost, still missing | 11 |
| 🆕 Proposed this revision | 32 |
| ↺ Rows added for already-built features | 5 |

### Coverage by domain

| Domain | Rows | Done | 🆕 New |
|--------|------|------|--------|
| 1. Team Offensive Efficiency | 19 | 11 | 1 |
| 2. Team Defensive Efficiency | 17 | 9 | 1 |
| 3. Turnover & Discipline | 9 | 3 | 1 |
| 4. Quarterback Quality | 16 | 0 | 3 |
| 5. Schedule & Situational Context | 20 | 13 | 4 |
| 6. Weather & Environment | 9 | 4 | 0 |
| 7. Market-Derived Features | 11 | 1 | 2 |
| 8. Player-Level Features | 29 | 10 | 5 |
| 9. Roster & Personnel | 14 | 0 | 1 |
| 10. Coaching & Scheme | 8 | 0 | 0 |
| 11. Historical Trends | 13 | 4 | 2 |
| 12. Special Teams & Field Position | 5 | 0 | 5 |
| 13. Unit Matchups & Interactions | 3 | 0 | 3 |
| 14. Rating-System Outputs | 6 | 2 | 4 |

---

## Glossary

*Plain-language definitions for terms used above.*

| Term | Meaning |
|------|---------|
| aDOT | Average depth of target: how far downfield a player's targets travel |
| ATS | Against the spread: whether a team covered the betting line |
| Brier score | Average squared error of probability forecasts; lower is better |
| Calibration / ECE | Whether "70%" predictions come true about 70% of the time; ECE (expected calibration error) summarizes the gap |
| CLV | Closing line value: how the price you took compares with the final pre-game price |
| Conformal prediction | A wrapper that turns any model's output into intervals with guaranteed coverage |
| CPOE | Completion percentage over expected, given how hard each throw was |
| DVOA | FTN's efficiency measure, adjusted for opponent and situation |
| EPA | Expected points added: how much a single play changed the offense's expected points |
| Garbage time | Plays after the outcome is effectively decided |
| Glicko-2 / RD | A rating system like Elo that also tracks uncertainty; RD (rating deviation) is that uncertainty |
| GNN | Graph neural network: a model that learns from relationships between players on the field |
| Leakage | Information from the future, or from the outcome itself, sneaking into features |
| Log loss | A probability score that heavily penalizes confident wrong predictions |
| NGS | Next Gen Stats: the NFL's player-tracking system |
| No-vig probability | A win probability from betting odds with the bookmaker's margin removed |
| OpenSkill (μ, σ, τ) | A multi-player rating system: μ is estimated skill, σ is uncertainty, τ keeps ratings from freezing |
| PBP | Play-by-play data |
| PROE | Pass rate over expected: how much more (or less) a team passes than typical in the same situation |
| Pythagorean win % | Expected win percentage from points scored and allowed |
| RAPM / HBPM | Plus-minus methods that credit players for team results while adjusting for teammates and opponents |
| Replacement level | What a readily available backup would produce |
| RYOE | Rush yards over expected, given blockers and defenders at the handoff |
| SHAP | A method that shows how much each feature pushed a specific prediction up or down |
| SOS | Strength of schedule |
| SRS | Simple Rating System: point margin adjusted for opponents |
| Success rate | Share of plays with positive EPA |
| Walk-forward validation | Train on the past, test on the next period, then roll forward |
| WAR | Wins above replacement |
| WHR | Whole History Rating: re-estimates all past ratings at once instead of updating game by game |
| WP / WPA | Win probability / win probability added by a play |
| YAC | Yards after catch |

---

## References

**Rating systems and ranking methods**

- Bradley, R. A. & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*.
- Colley, W. N. (2002). Colley's Bias Free College Football Ranking Method. https://www.colleyrankings.com/matrate.pdf
- Coulom, R. (2008). Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength. https://www.remi-coulom.fr/WHR/
- Glickman, M. E. The Glicko-2 system (glicko.net).
- Keener, J. P. (1993). The Perron–Frobenius Theorem and the Ranking of Football Teams. *SIAM Review*. https://www2.math.upenn.edu/~kazdan/312S14/Notes/Perron-Frobenius-football-SIAM1993.pdf
- Minka, T., Cleven, R. & Zaykov, Y. (2018). TrueSkill 2: An improved Bayesian skill rating system. Microsoft Research.
- OpenSkill documentation: https://openskill.me/en/stable/
- TrueSkill (Python package and licensing notes): https://trueskill.org/
- PandaSkill (2025): performance scores feeding an OpenSkill free-for-all rating, applied to League of Legends.

**Player value and football analytics**

- Nguyen, Q., Yurko, R. & Matthews, G. J. (2023). Here Comes the STRAIN: Analyzing Defensive Pass Rush in American Football with Player Tracking Data. https://arxiv.org/pdf/2305.10262
- Yurko, R., Ventura, S. & Horowitz, M. (2019). nflWAR: A Reproducible Method for Offensive Player Evaluation in Football. *Journal of Quantitative Analysis in Sports*.
- Smith, R. S., Guilleminault, C. & Efron, B. (1997). Circadian rhythms and enhanced athletic performance in the National Football League. *Sleep*.
- NFL Big Data Bowl: https://operations.nfl.com/gameday/analytics/big-data-bowl

**Data**

- nflreadpy load functions: https://nflreadpy.nflverse.com/api/load_functions/
- nflverse data update schedule: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html
- FTN (DVOA): ftnfantasy.com

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-25 | Adopted the game-model rows of Priority Matrix v2 into a "Game-Model Build Queue," mapped to ROADMAP.md Tier 5 #16 (game models only; prop rows stay proposed). Corrected 2 Status values from ❌ to ⚠️ Partial (Season week number, Indoor override — both already partially built) and closed the duplicate "Pace tendency" row (D10) against D1's "Plays per game / pace." Corrected the expanded feature-set count from "149" (never accurate) to `expanded_152`, the current registered name. Recorded that the Part II.5 chronological-splits prerequisite is already satisfied. No other Status, Signal, Cost, or Model Target values changed. |
| 2026-09-24 | Merged the modeling reference into this file (Part II: methods, rating systems, validation, construction rules, data sources) and retitled it Feature Catalog & Modeling Reference. Added Domains 12–14 and 32 🆕 proposed features across the catalog; added 5 ↺ rows for built features that had no row (touch share, avg_score_diff, close_game_pct, team Elo, SOS via opponent Elo). Corrected 16 Cost ratings and updated 33 descriptions (itemized below). No Status, Signal, or Model Target values changed. Priority Matrix v1 kept for history; v2 proposed. Exact counts replace the earlier approximate summary: the 2026-06-10 catalog had 143 rows (52 done, 1 partial, 90 missing). |
| 2026-06-10 | W4 player features built. Marked ~15 Domain 8 features as Done: rolling stats (L3+L6 mean/std), usage shares (target/carry/touch), matchup ranks, game context (spread, total, dome, home, rest, implied team total). Priority Matrix items 8–10, 12 complete. Snap % deferred (nflreadpy doesn't expose snap counts). |
| 2026-06-04 | Marked plays/pace, yards_per_play, redzone_attempts, int_rate (off), penalty_rate, avg_score_diff, close_game_pct as DONE. CPOE marked Partial (computed but excluded from model features due to NaN). EPA_COLS 22→36, _EXPANDED_FEATURES 107→149. Champions rejected challengers - features retained for future prop models and systematic selection. |
| 2026-06-01 | Marked Phase 20e priorities 1–7, 14–15 as DONE. Added 14 features across EPA, efficiency, and situational domains. |
| 2026-05-30 | Initial version - comprehensive brainstorm from prototype review + gap analysis. |

### 2026-09-24 edit log

**Cost corrections** (the data already sits in PBP, schedules, or nflreadpy):

- Med → Low: Pass rate (neutral script), Air yards / attempt, YAC / completion (D1); QB EPA/play (career), QB EPA/play (rolling L4–L6), Scramble rate, Designed rush rate (D4); Consensus closing spread, Consensus closing total, Market win probability (no-vig) (D7); Head coach win % (D10); ATS record, Over/Under record (D11).
- High → Low: CPOE (D1), CPOE (rolling) (D4).
- High → Med: Pressure rate (D2).

**Description updates** (original wording kept, notes appended unless noted):

- D1: Pass rate (neutral script) → PROE via `xpass`/`pass_oe`; CPOE → NaN fix; Air yards and YAC → PBP column names.
- D2: Pressure rate → participation and PFR sources plus the calibration caveat (replaced "(requires charting data)"); Def DVOA → FTN as publisher, pointer to the DIY row (replaced "Football Outsiders adjusted metric").
- D4: QB Elo → Glicko-2/OpenSkill alternatives; QB EPA (career), CPOE (rolling), Scramble rate, Designed rush rate → PBP columns; QB change flag → schedules source and the projected-starter caveat.
- D5: Home field advantage → re-estimate per season if coefficients are fixed.
- D6: Weather → feature wiring (replaced "isn't wired into prediction features yet," which contradicted its ✅); Indoor override → retractable roofs. Timing note added below the table.
- D7: Consensus closing spread and total → already ingested for Props, in schedules; Opening line and Line movement → odds-ledger dependency; No-vig → moneylines in schedules. Timing note added below the table.
- D8: Snap % → unblocked via `load_snap_counts()` plus ID crosswalk; Route participation rate → still needs paid charting.
- D9: Player WAR, On/off EPA split, Backup quality rating → Part II method pointers; Injury status → ESPN source switch and the 2016 retirement of "Probable."
- D10: Head coach win % and New coaching staff flag → schedules source; Play-calling tendency and Pace tendency → overlap and duplicate flags.
- D11: Win streak → momentum caveat; ATS record and Over/Under record → schedules source.
- Domain 9 What-If engine block labeled illustrative; method hooks added.

# Gridiron Edge - Feature Catalog

## Comprehensive inventory of model features: existing, planned, and aspirational

---

## How This Document Fits

| Document | Relationship |
|----------|-------------|
| **FEATURES.md** (this file) | Living reference of all features across all domains. Updated as features are built, added, or deprioritized. |
| **ROADMAP.md** | References feature domains at the workstream level. Links here for detail. |
| **PLAN.md** | Pulls specific features from the Priority Matrix into short-term task lists. |
| **CHANGELOG.md** | Records when features move from Missing to Done. |

### Column definitions

| Column | Meaning |
|--------|---------|
| **Model Target** | Which model(s) this feature feeds: **Game** (M2), **Props** (M3), **What-If** (M4/scenario), **Market** (M5) |
| **Signal** | Estimated predictive value: 🔴 High / 🟡 Med / 🟢 Low / ❓ Unknown |
| **Cost** | Implementation effort: Low (data exists, simple transform) / Med / High (new data source or complex engineering) |
| **Status** | ✅ Done / ⚠️ Partial / ❌ Missing |

---

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
| Pass rate (neutral script) | Pass-heavy when game is close? | Props | 🟡 Med | Med | ❌ |
| Pass rate (overall) | Raw pass/run ratio | Props | 🟢 Low | Low | ❌ |
| Yards per play | Simpler efficiency proxy | Game | 🟢 Low | Low | ✅ Done |
| CPOE | Completion % over expected | Game, Props | 🔴 High | High | ⚠️ Partial |
| Air yards / attempt | Depth of target proxy | Props | 🟡 Med | Med | ❌ |
| YAC / completion | Yards after catch - scheme/personnel signal | Props | 🟡 Med | Med | ❌ |

---

## Domain 2: Team Defensive Efficiency

*How well does this team prevent the opponent from doing things?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Def EPA/play (overall) | Defensive expected points allowed per play | Game, Props | 🔴 High | Low | ✅ Done |
| Def EPA/play (pass) | Against the pass | Game, Props | 🔴 High | Low | ✅ Done |
| Def EPA/play (rush) | Against the run | Game, Props | 🔴 High | Low | ✅ Done |
| Def success rate | % of opponent plays held to negative EPA | Game | 🟡 Med | Low | ✅ Done |
| Pressure rate | QB pressures / dropbacks (requires charting data) | Game, Props | 🔴 High | High | ❌ |
| Sack rate | Sacks / dropbacks | Game, Props | 🟡 Med | Low | ✅ Done |
| Rush yards allowed / game | Volume stat, useful for prop matchups | Props | 🟡 Med | Low | ❌ |
| Pass yards allowed / game | Volume stat | Props | 🟡 Med | Low | ❌ |
| Opponent 3rd down conversion % | Defensive 3rd down stops | Game | 🟡 Med | Low | ✅ Done |
| Opponent red zone TD % | Bending but not breaking? | Game | 🟡 Med | Low | ✅ Done |
| Explosive plays allowed rate | Big play vulnerability | Game | 🟡 Med | Low | ✅ Done |
| Turnover creation rate | Forced fumbles + INTs per game | Game | 🟡 Med | Low | ✅ Done |
| Opponent completion % | Raw passing defense | Props | 🟢 Low | Low | ❌ |
| Def DVOA (if sourced) | Football Outsiders adjusted metric | Game | 🔴 High | High | ❌ |
| Points allowed / game | Simple but noisy | Game | 🟢 Low | Low | ❌ |
| Opponent QB rating | Passer rating allowed | Props | 🟡 Med | Med | ❌ |

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

---

## Domain 4: Quarterback Quality

*The single most important position. Deserves its own feature domain.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| QB Elo / QB-specific rating | Separate Elo track for the starting QB | Game | 🔴 High | High | ❌ |
| QB EPA/play (career) | Baseline quality signal | Game, Props | 🔴 High | Med | ❌ |
| QB EPA/play (rolling L4–L6) | Recent form | Game, Props | 🔴 High | Med | ❌ |
| Passer rating (rolling) | Traditional metric, still informative | Props | 🟡 Med | Low | ❌ |
| Completion % (rolling) | Raw and rolling | Props | 🟡 Med | Low | ❌ |
| CPOE (rolling) | Accuracy over expectation | Game, Props | 🔴 High | High | ❌ |
| Sack rate taken | How often the QB gets sacked | Props | 🟡 Med | Low | ❌ |
| Scramble rate | How often the QB runs when play breaks down | Props | 🟡 Med | Med | ❌ |
| Designed rush rate | Designed QB runs per game - key for rushing props | Props | 🟡 Med | Med | ❌ |
| QB rush yards / game (rolling) | Direct input for QB rush prop models | Props | 🔴 High | Low | ❌ |
| QB pass yards / game (rolling) | Direct input for QB pass prop models | Props | 🔴 High | Low | ❌ |
| QB change flag | Is a different QB starting than last week? | Game, What-If | 🔴 High | Med | ❌ |
| QB experience (games started) | Rookie vs veteran signal | Game | 🟢 Low | Low | ❌ |

---

## Domain 5: Schedule & Situational Context

*When, where, and under what circumstances is the game played?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Home field advantage | Binary + strength | Game | 🔴 High | Low | ✅ Done |
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
| Season week number | Early vs late season dynamics | Game | 🟢 Low | Low | ❌ |
| Playoff/elimination context | Must-win games may play differently | Game | 🟢 Low | Med | ❌ |
| Rest differential | Team A days rest minus Team B days rest | Game | 🟡 Med | Low | ✅ Done |
| Opponent rest | The other team's rest situation | Game | 🟡 Med | Low | ✅ Done |
| Back-to-back road games | Fatigue / travel compounding | Game | 🟢 Low | Low | ❌ |

---

## Domain 6: Weather & Environment

*Physical conditions that affect play style and outcomes.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Temperature (F) | Cold weather affects passing, grip | Game, Props | 🟡 Med | Low | ✅ Done |
| Wind speed (mph) | Affects kicking, deep passing | Game, Props | 🟡 Med | Low | ✅ Done |
| Precipitation flag | Rain/snow binary | Game, Props | 🟡 Med | Low | ✅ Done |
| Weather → feature wiring | OWM data exists but isn't wired into prediction features yet | Game, Props | 🟡 Med | Low | ✅ Done |
| Wind speed bins | Calm (0–10), moderate (10–20), high (20+) | Game, Props | 🟡 Med | Low | ❌ |
| Cold weather flag | Below 32°F threshold | Props | 🟡 Med | Low | ❌ |
| Indoor override | If dome, weather features zeroed out | Game, Props | 🟡 Med | Low | ❌ |
| Precipitation type | Rain vs snow (different effects) | Game | 🟢 Low | Med | ❌ |
| Historical weather impact | Team's performance split by weather bucket | Game | 🟢 Low | Med | ❌ |

---

## Domain 7: Market-Derived Features

*What does the betting market itself tell us?*

> **Philosophical note:** Market features are extremely powerful but create a tension. Using the closing line as a feature means your model is partially *following* the market rather than *disagreeing* with it. This is fine for total projection accuracy but can mask whether your non-market features have genuine signal. **Recommendation:** Train both a "market-aware" and "market-blind" model variant and compare.

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Consensus closing spread | The market's best estimate of team strength | Game | 🔴 High | Med | ❌ |
| Consensus closing total | Market estimate of combined scoring | Game, Props | 🔴 High | Med | ❌ |
| Opening line | Where the line opened - often reflects sharp money | Game | 🟡 Med | Med | ❌ |
| Line movement (open → current) | Direction and magnitude of movement | Game | 🟡 Med | Med | ❌ |
| Implied team total | (Total ± Spread) / 2 - crucial for prop context | Props | 🔴 High | Low | ✅ Done |
| Market win probability (no-vig) | De-vigged implied probability from market | Game | 🔴 High | Med | ❌ |
| Reverse line movement flag | Line moves opposite to public money | Game | 🟡 Med | High | ❌ |
| Sharp book (Pinnacle) line | Pinnacle as a separate "sharp" feature | Game | 🟡 Med | Med | ❌ |
| Historical closing line (as prior) | Use last season's closing lines as team-quality prior early in season | Game | 🟡 Med | Med | ❌ |

---

## Domain 8: Player-Level Features (for Prop Models)

*Individual player performance, usage, and context.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Rolling stat mean (L3, L6, L12) | Per-stat rolling averages at multiple windows | Props | 🔴 High | Med | ✅ Done (L3 + L6) |
| Rolling stat median | More robust to outliers than mean | Props | 🟡 Med | Med | ❌ |
| Rolling stat std dev | Player's own variance - feeds uncertainty bands | Props | 🔴 High | Med | ✅ Done |
| Season average | Full-season baseline | Props | 🟡 Med | Low | ❌ |
| Snap % (rolling) | Playing time trend | Props | 🔴 High | Med | ❌ |
| Target share (WR/TE) | % of team targets | Props | 🔴 High | Med | ✅ Done |
| Carry share (RB) | % of team carries | Props | 🔴 High | Med | ✅ Done |
| Route participation rate | % of pass plays where WR runs a route | Props | 🟡 Med | High | ❌ |
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

---

## Domain 9: Roster & Personnel - The What-If Domain

*Who is playing, who isn't, and what does it mean?*

This is the domain that powers the **Scenario Engine** (ROADMAP W4.5). It's the most complex and the most differentiating capability.

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Player WAR (wins above replacement) | How many wins does this player add? | What-If, Game | 🔴 High | High | ❌ |
| On/off EPA split | Team EPA with vs without this player | What-If, Game | 🔴 High | High | ❌ |
| Positional importance weight | QB > Edge > WR > RB for team impact | What-If, Game | 🟡 Med | Med | ❌ |
| Injury status (Out/Doubtful/Q/Probable) | Current game status | What-If, Game, Props | 🔴 High | Med | ❌ |
| Estimated play probability | Probability player actually plays given status | What-If | 🟡 Med | Med | ❌ |
| Backup quality rating | How good is the replacement? | What-If, Game | 🔴 High | High | ❌ |
| Usage redistribution model | If RB1 out → RB2 gets X% of carries, RB3 gets Y% | What-If, Props | 🔴 High | High | ❌ |
| Target tree redistribution | If WR1 out → WR2/TE1/RB target shares shift | What-If, Props | 🔴 High | High | ❌ |
| Cumulative injury impact | Sum of WAR-weighted absences on a roster | What-If, Game | 🔴 High | High | ❌ |
| O-line health index | Composite of OL starters available | Game, Props | 🟡 Med | High | ❌ |
| Historical with/without record | Team's ATS record with and without this player | What-If | 🟡 Med | Med | ❌ |
| Depth chart stability | How much roster churn has occurred recently | Game | 🟢 Low | Med | ❌ |
| Games together (unit cohesion) | How many games has the current OL/WR corps played together | Game | 🟢 Low | High | ❌ |

### How the What-If Scenario Engine Would Work

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

---

## Domain 10: Coaching & Scheme

*How does the coaching staff affect game outcomes and player usage?*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Head coach win % (career) | Baseline coaching quality | Game | 🟢 Low | Med | ❌ |
| HC tenure (years with team) | Stability / familiarity | Game | 🟢 Low | Low | ❌ |
| Offensive coordinator tenure | New OC = scheme adjustment period | Game, Props | 🟡 Med | Med | ❌ |
| Play-calling tendency (pass/run) | Coaching scheme signature | Props | 🟡 Med | Med | ❌ |
| Pace tendency (plays/game) | Fast vs slow teams | Props | 🟡 Med | Low | ❌ |
| Aggressiveness (4th down go rate) | Risk tolerance proxy | Game | 🟢 Low | Med | ❌ |
| Historical ATS performance | Some coaches consistently beat/miss spreads | Game | 🟢 Low | Med | ❌ |
| Coaching matchup history | H2H coaching record | Game | 🟢 Low | Med | ❌ |
| New coaching staff flag | First-year HC/OC/DC | Game | 🟡 Med | Low | ❌ |

---

## Domain 11: Historical Trends & Situational Patterns

*Meta-features about when/how teams perform differently.*

| Feature | Description | Model Target | Signal | Cost | Status |
|---------|-------------|-------------|--------|------|--------|
| Win streak / loss streak | Momentum proxy | Game | 🟢 Low | Low | ✅ Done |
| Win % (season) | Current season record | Game | 🟢 Low | Low | ✅ Done |
| ATS record (season) | Betting-specific performance | Game | 🟡 Med | Med | ❌ |
| Over/Under record (season) | Tendency to go over or under | Game | 🟡 Med | Med | ❌ |
| Performance after loss | Bounce-back tendency | Game | 🟢 Low | Low | ❌ |
| Performance as favorite/underdog | Does the team cover more as dog or fav? | Game | 🟡 Med | Med | ❌ |
| Performance by spread bucket | ATS record in -3 to -7 range vs -7+ etc. | Game | 🟢 Low | Med | ❌ |
| Scoring by quarter | Is the team a fast or slow starter? | Game | 🟢 Low | Med | ❌ |
| Season week performance | Early vs mid vs late season form | Game | 🟢 Low | Low | ❌ |

---

## Priority Matrix: Top 15 Features to Add Next

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

Items 1–7 and 14–15 are complete (feature engineering done, 149 features).
Items 8–13 are W4 player data (start once player game logs are ingested).

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total features cataloged | ~120 |
| Domains | 11 |
| Currently Done | ~45+ |
| Partial / In Schema | ~6 |
| Missing | ~75 |
| High-signal features | ~35 |
| Low-cost features | ~50 |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-10 | W4 player features built. Marked ~15 Domain 8 features as Done: rolling stats (L3+L6 mean/std), usage shares (target/carry/touch), matchup ranks, game context (spread, total, dome, home, rest, implied team total). Priority Matrix items 8–10, 12 complete. Snap % deferred (nflreadpy doesn't expose snap counts). |
| 2026-06-04 | Marked plays/pace, yards_per_play, redzone_attempts, int_rate (off), penalty_rate, avg_score_diff, close_game_pct as DONE. CPOE marked Partial (computed but excluded from model features due to NaN). EPA_COLS 22→36, _EXPANDED_FEATURES 107→149. Champions rejected challengers - features retained for future prop models and systematic selection. |
| 2026-06-01 | Marked Phase 20e priorities 1–7, 14–15 as DONE. Added 14 features across EPA, efficiency, and situational domains. |
| 2026-05-30 | Initial version - comprehensive brainstorm from prototype review + gap analysis. |

# Projected starting-QB backtest

Generated 2026-10-03. Read-only evaluation (`PLAN.md` Unit 9); no model, schema or `data/` artifact changed.

## Question

Can a rules-based projected starter, built from the depth chart and the weekly availability status, name the quarterback who actually starts more often than carrying forward each team's previous starter (`DECISIONS.md` D65)?

Scored: 8,309 regular-season team-games, 2009-2024, with a depth chart and a previous starter. The injury report is the last report of the week, so every reading below is the state at the final weekly report.

## Result on every week

| Rule | Accuracy | Gain vs carry-forward (points) | 90% interval | Seasons won |
|---|---|---|---|---|
| Carry-forward (previous starter, D65) | 88.4% | baseline | | |
| Depth chart QB1 | 89.8% | +1.36 | [+0.84, +1.87] | 87.5% |
| Primary: QB1 to QB3, skipping Out and Doubtful | 92.7% | +4.30 | [+3.79, +4.85] | 100.0% |
| Variant: also skipping Questionable (information only) | 92.2% | +3.79 | [+3.20, +4.39] | 100.0% |

## Leakage guard

Week 1 accuracy of the primary rule is 99.2% (506 games) against 92.3% afterwards: a gap of 6.9 points. The guard fires above 5 points, so it **fired**. Week 1 is excluded from the decision and both readings are reported.

Carry-forward is weak in week 1 only because offseason moves change starters (58.3% correct), so a large week-1 gain is expected even from a genuine pre-game chart. The 2025 snapshot test below checks week 1 directly.

### Weeks after week 1 (the decision reading when the guard fires)

| Rule | Accuracy | Gain vs carry-forward (points) | 90% interval | Seasons won |
|---|---|---|---|---|
| Carry-forward (previous starter, D65) | 90.3% | baseline | | |
| Depth chart QB1 | 89.2% | -1.18 | [-1.64, -0.72] | 12.5% |
| Primary: QB1 to QB3, skipping Out and Doubtful | 92.3% | +1.92 | [+1.44, +2.42] | 93.8% |
| Variant: also skipping Questionable (information only) | 91.8% | +1.42 | [+0.87, +1.99] | 75.0% |

## Bar

Pre-declared: the primary rule's accuracy exceeds carry-forward by at least 2.0 points, the 90% interval excludes zero, and it wins in at least 70% of seasons.

- Every week: gain +4.30 points, interval [+3.79, +4.85], seasons won 100.0%.
- Weeks after week 1: gain +1.92 points, interval [+1.44, +2.42], seasons won 93.8%.
- Decision reading: weeks after week 1.

**Bar not cleared.**

## Remaining errors of the primary rule

- Misses: 607
- The actual starter was in the top three of the chart: 85.0%
- The actual starter was the carried-forward quarterback the rule skipped: 31.6%
- Misses in week 1: 0.7%

## By season (primary rule, gain over carry-forward in points)

| Season | Games | Carry-forward | Primary | Gain |
|---|---|---|---|---|
| 2009 | 512 | 89.3% | 93.0% | +3.71 |
| 2010 | 511 | 85.9% | 91.0% | +5.09 |
| 2011 | 510 | 90.4% | 93.3% | +2.94 |
| 2012 | 512 | 92.2% | 95.1% | +2.93 |
| 2013 | 512 | 90.6% | 94.5% | +3.91 |
| 2014 | 512 | 89.8% | 96.1% | +6.25 |
| 2015 | 510 | 88.0% | 94.1% | +6.08 |
| 2016 | 511 | 89.6% | 94.5% | +4.89 |
| 2017 | 511 | 88.6% | 92.2% | +3.52 |
| 2018 | 512 | 90.4% | 93.8% | +3.32 |
| 2019 | 512 | 89.3% | 92.2% | +2.93 |
| 2020 | 511 | 87.5% | 93.7% | +6.26 |
| 2021 | 544 | 84.9% | 90.1% | +5.15 |
| 2022 | 542 | 85.4% | 90.8% | +5.35 |
| 2023 | 543 | 86.0% | 90.1% | +4.05 |
| 2024 | 544 | 86.9% | 89.3% | +2.39 |

## Timing evidence (2025, descriptive only)

Latest depth chart snapshot on or before the lead time. `Chart only` is fully as-of. `Chart and report` adds the week's final injury report; the 2025 injury file has no modification time, so that column is optimistic about what was known at the lead time. No bar applies.

### All weeks

| Lead (hours) | Team-games | Chart only | Chart and report |
|---|---|---|---|
| 24 | 544 | 90.6% | 95.8% |
| 48 | 544 | 90.6% | 95.6% |
| 96 | 544 | 90.1% | 94.9% |

### Week 1

| Lead (hours) | Team-games | Chart only | Chart and report |
|---|---|---|---|
| 24 | 32 | 100.0% | 100.0% |
| 48 | 32 | 100.0% | 100.0% |
| 96 | 32 | 100.0% | 100.0% |

### Weeks after week 1

| Lead (hours) | Team-games | Chart only | Chart and report |
|---|---|---|---|
| 24 | 512 | 90.0% | 95.5% |
| 48 | 512 | 90.0% | 95.3% |
| 96 | 512 | 89.5% | 94.5% |

## Caveats

- Depth charts before 2025 carry no timestamp, so pre-kickoff capture is inferred from accuracy, not verified.
- Injury reports are the last report of the week (mostly Friday). The rule therefore needs information that exists late in the week.
- Starters change for reasons an injury report never shows (benching, rest, late scratches on game day).

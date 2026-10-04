# Quote movement and the market age limit

Generated 2026-10-04 by `gridiron evaluate quote-movement` (`DECISIONS.md` D80).

## Measure

The sportsbooks' de-vigged moneyline probability for the home team, from the quote
ledger (pregame quotes only). The consensus is the median across books at one
moment. *Disagreement* is how far one book sits from the consensus at one moment.
*Drift* is how far the consensus moves between two snapshots of the same game, by the
age between them. Both are in percentage points of win probability; at even odds an
expected-value change is twice this.

## Data

48 games; snapshots per week: week 1: 34, week 2: 1, week 3: 12, week 4: 10. Pairs of snapshots of one game overlap, so they are not independent observations.

## Disagreement between books at one moment

Median distance from the consensus 0.34 points (90th percentile 1.08); median spread between the highest and lowest book 1.81 points.

## Drift by age

| Age between snapshots (h) | Pairs | Median drift (points) | 90th percentile |
|---|---|---|---|
| 0-1 | 79 | 0.00 | 0.33 |
| 1-3 | 483 | 0.00 | 0.40 |
| 3-6 | 411 | 0.10 | 0.74 |
| 6-12 | 650 | 0.17 | 0.94 |
| 12-24 | 556 | 0.28 | 0.87 |
| 24-48 | 579 | 0.34 | 1.60 |
| 48-72 | 764 | 0.97 | 1.97 |
| 72-∞ | 1185 | 1.28 | 2.84 |

## Age limit

Rule: the limit is the lower edge of the first age bin whose median drift exceeds the
median disagreement between books, requiring at least 30 pairs per bin. A quote older than that has moved by more than the
books differ among themselves.

**Derived by the rule:** 48 hours (median drift 0.97 points at 48-72 h exceeds the 0.34-point disagreement between books).
**Taken (borderline rule):** 24 hours. The bin below the derived limit has a median within 0.05 points of the disagreement, so the lower bin edge is taken: blocking early only delays an edge, while passing a stale quote can publish a false one.
**Configured (`MAX_MARKET_AGE`):** 24 hours (matches).

## How this limit was chosen, and what it is not

The rule was written after reading the first dense weeks of the ledger (week 1 from
the original collector, then weeks 3 and 4), so it is a post-hoc criterion, not a
pre-declared one; the weeks collected from now on are the out-of-sample check, and this
command should be rerun as they arrive. The bins are coarse: the derived limit is a bin
edge, and the median drift just above it is close to the disagreement. It is a single
age for every market, derived from moneylines; spreads and totals move less.

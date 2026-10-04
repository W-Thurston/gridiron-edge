# Does the model beat the closing market where they disagree?

Generated 2026-10-04 by `gridiron evaluate market-agreement` (`DECISIONS.md` D85).

## Measure

Walk-forward backfilled forecasts, 2013 onward, against the closing lines in the cleaned games file. For win probability the market is the closing spread's implied probability (`VEGAS_LINE` is the expected home margin; a normal margin with standard deviation 13.5); for totals it is the closing over/under. The *gap* is model minus market. The *slope* is how much of the gap shows up in the outcome (fitted through the origin, robust standard error): 0 means the market is right and the disagreement is noise, 1 means the model is right.

**Rule (recorded before it was applied):** raw model EV is trustworthy only if the slope's 95% interval excludes 0 and includes 1. The gap bins and the reading of `VEGAS_LINE` were chosen after seeing the data. Ties and pushes are excluded.

## Summary

| Family | Games | Model | Market | Slope | 95% interval | Validated |
|---|---|---|---|---|---|---|
| Win - Elo | 3598 | Brier 0.2305 | Brier 0.2119 | -0.146 | -0.263 to -0.029 | no |
| Win - Logistic | 3550 | Brier 0.2198 | Brier 0.2117 | 0.017 | -0.152 to 0.187 | no |
| Win - Random Forest | 3550 | Brier 0.2203 | Brier 0.2117 | 0.039 | -0.119 to 0.196 | no |
| Win - XGBoost | 3550 | Brier 0.2217 | Brier 0.2117 | -0.019 | -0.173 to 0.134 | no |
| Total - Random Forest | 3562 | MAE 10.7567 | MAE 10.5053 | 0.075 | -0.067 to 0.217 | no |
| Total - XGBoost | 3562 | MAE 10.7550 | MAE 10.5053 | 0.115 | -0.023 to 0.253 | no |

## Win - Elo

| Model minus market | Games | Mean model | Mean market | Realized home win |
|---|---|---|---|---|
| -1 to -0.1 | 1277 | 0.524 | 0.688 | 0.710 |
| -0.1 to -0.05 | 633 | 0.525 | 0.600 | 0.597 |
| -0.05 to +0.05 | 1031 | 0.480 | 0.484 | 0.482 |
| +0.05 to +0.1 | 354 | 0.458 | 0.384 | 0.370 |
| +0.1 to +1 | 303 | 0.468 | 0.318 | 0.284 |

## Win - Logistic

| Model minus market | Games | Mean model | Mean market | Realized home win |
|---|---|---|---|---|
| -1 to -0.1 | 335 | 0.458 | 0.607 | 0.600 |
| -0.1 to -0.05 | 457 | 0.529 | 0.601 | 0.597 |
| -0.05 to +0.05 | 1567 | 0.569 | 0.568 | 0.580 |
| +0.05 to +0.1 | 629 | 0.599 | 0.526 | 0.533 |
| +0.1 to +1 | 562 | 0.621 | 0.469 | 0.450 |

## Win - Random Forest

| Model minus market | Games | Mean model | Mean market | Realized home win |
|---|---|---|---|---|
| -1 to -0.1 | 370 | 0.475 | 0.627 | 0.603 |
| -0.1 to -0.05 | 401 | 0.517 | 0.589 | 0.606 |
| -0.05 to +0.05 | 1502 | 0.559 | 0.559 | 0.559 |
| +0.05 to +0.1 | 642 | 0.616 | 0.543 | 0.573 |
| +0.1 to +1 | 635 | 0.637 | 0.484 | 0.468 |

## Win - XGBoost

| Model minus market | Games | Mean model | Mean market | Realized home win |
|---|---|---|---|---|
| -1 to -0.1 | 374 | 0.471 | 0.624 | 0.615 |
| -0.1 to -0.05 | 465 | 0.542 | 0.614 | 0.637 |
| -0.05 to +0.05 | 1432 | 0.571 | 0.569 | 0.573 |
| +0.05 to +0.1 | 633 | 0.594 | 0.520 | 0.506 |
| +0.1 to +1 | 646 | 0.622 | 0.466 | 0.471 |

## Total - Random Forest

| Model minus line (points) | Games | Mean gap | Mean actual minus line |
|---|---|---|---|
| ∞ to -6 | 145 | -7.31 | -0.47 |
| -6 to -4 | 264 | -4.93 | -0.02 |
| -4 to -2 | 566 | -2.89 | +0.42 |
| -2 to +2 | 1721 | +0.04 | +0.28 |
| +2 to +4 | 548 | +2.86 | +0.29 |
| +4 to +6 | 237 | +4.77 | +1.15 |
| +6 to ∞ | 81 | +7.22 | +3.54 |

## Total - XGBoost

| Model minus line (points) | Games | Mean gap | Mean actual minus line |
|---|---|---|---|
| ∞ to -6 | 176 | -7.34 | -0.80 |
| -6 to -4 | 266 | -4.87 | +0.96 |
| -4 to -2 | 604 | -2.90 | +0.41 |
| -2 to +2 | 1650 | +0.05 | -0.02 |
| +2 to +4 | 538 | +2.91 | +0.41 |
| +4 to +6 | 234 | +4.80 | +2.47 |
| +6 to ∞ | 94 | +7.16 | +2.35 |

## Verdict

No family passes the rule: where the model disagrees with the closing market, outcomes follow the market. Raw model EV is not an edge.

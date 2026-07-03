# Gridiron Edge — Frontend Design Decisions

Short reference for how the frontend applies the OKLCH dark theme and
typography from the prototype. Adds context that would otherwise be
lost when reading the JSX in isolation.

## Aesthetic identity: functional blend

The prototype README described three optional aesthetic variants —
Terminal, Fintech, Editorial. W9 doesn't formally pick one. Instead
we ship a blend, driven by function:

- **Cards** (`.hm-card`) for grouping related content — Fintech-adjacent.
- **Monospace numerics** (`.mono .tnum`) for anywhere digits need to
  align vertically — Terminal-adjacent. Applies to: odds, stakes,
  P&L, ratings, probabilities, percentages, timestamps.
- **Serif titles** (`.serif`) for chapter-like emphasis — Editorial-adjacent.
  Applies to: blocked-screen titles, onboarding wizard step headers.
- **Sans (Geist)** everywhere else — body text, form labels, buttons.

The identity is "Bloomberg terminal for football." Dense, tabular,
data-forward. That comes through composition, not from committing to
any single variant.

## When to reach for each

**Serif (Instrument Serif):**
- Chapter titles on standalone screens (BlockedScreen, Onboarding).
- One-word screen names when they open a new context.
- Never for tabular data or numerics.

**Monospace (Geist Mono):**
- Every table cell that contains a number.
- Odds, stakes, probabilities, percentages.
- Timestamps.
- Any tabular column-aligned display.
- Small pill labels ("+150", "WON").

**Sans (Geist):**
- Body text, paragraphs.
- Form labels, buttons.
- Nav items.
- Player names, team names (when not in a table cell).

## Color semantics

Tokens from `src/index.css`:

- `--pos` — favorable outcomes: winning bets, positive P&L, positive
  EV, upward trends, active states.
- `--neg` — adverse outcomes: losing bets, negative P&L, blockers,
  destructive actions.
- `--warn` — moderate/caution states: middling edge strength,
  uncertain projections.
- `--info` — neutral informational: badges, hints.
- `--ink-4` — placeholder/empty states.

## When you're unsure

Default to the blend that's already there. If a screen feels off,
it's usually because a numeric is in sans (should be mono) or a
paragraph is in mono (should be sans). Don't invent new variants.

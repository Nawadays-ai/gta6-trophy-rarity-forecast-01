# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nawadays-ai/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nawadays-ai/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A dependency-free Python program that forecasts how the earn rate of
each PlayStation trophy for GTA VI will move between a checkpoint page
snapshot and a future target time. For every trophy it emits five
quantiles (5/25/50/75/95) of the predicted earn percentage.

## Running it

The program is a single file, `main.py`, and needs nothing beyond the
Python 3 standard library.

- It scans the directory named by the `MAKEFUN_INPUT_DIR` environment
  variable (falling back to `/input`) for `*.in` files, in name order.
- Each input file is one JSON case: a trophy list, a history of page
  reads (timestamp, tracked-player count, per-trophy earn rates), the
  checkpoint time, the target time, the horizon in minutes, and the
  quantile levels to produce.
- For each case it writes `<case id>.out` into `MAKEFUN_OUTPUT_DIR`
  (falling back to `/output`), holding one JSON object with a
  `trophies` list of `{"id": ..., "q": [...]}` rows.

Run it with:

    python3 main.py

The output is deterministic: the same input always produces the same
bytes. Degenerate inputs (no history, a single page state, trophies
that were never observed) still yield a valid forecast through a
conservative fallback path instead of failing.

## How the forecast works

The model works on page states rather than raw reads: consecutive
reads that show the same player count and the same rates are merged
into a single state that keeps its first-seen time, and each state is
weighted by how long it stayed on display.

For every trophy, the earn rate is logit-transformed and regressed
against log time (ln of one plus the hours since the first state)
over the last 48 hours before the checkpoint, using a weighted least
squares fit whose weights are the square roots of the state
durations. A trophy needs at least three usable points with some
variation for its own fit; otherwise it borrows strength from its
rarity tier. Individual slopes are shrunk toward a tier pool, and the
pool itself is shrunk toward the field-wide median slope, so noisy
short histories stay close to what comparable trophies are doing.

The forecast is anchored on the last observed rate. The shrunk slope
moves that anchor toward the target, damped and capped so a single
trophy cannot leap implausibly far. The spread around the centre
combines the fit's residual scale, the slope's standard error grown
over the extrapolation distance, a floor term, and a random-walk term
that grows with the time since the last page change; trophies without
their own fit also pick up a term from the spread of observed slopes.
The interval is asymmetric near the bottom of the scale: as a centre
approaches the rarest rates, the lower side is squeezed and the upper
side widened.

Two consistency corrections finish the job. First, an earner floor:
players who already earned a trophy cannot un-earn it, so the
predicted rate never falls below the earners already counted divided
by a projection of the future player pool (a damped, capped
log-linear projection of the tracked-player count). Second, a
delayed-wave adjustment for silver and gold trophies that are still
rare while much of their cohort is stalled: their centre is pulled
part-way toward a 2 percent reference rate and the upper side is
widened, with a floor that keeps the lower side reachable. Finally,
the platinum row is capped level by level at the rarest of the other
rows whenever the last observed page itself respects that ordering,
since the platinum cannot be earned more often than the trophies it
requires.

Quantiles are read directly from the normal distribution at each
case's own levels — the centre plus the z-score times the appropriate
side's spread — then clamped to the valid rate range, rounded to four
decimals, and forced to be non-decreasing.

## Layout

- `main.py` — the whole program: input handling, the state
  reduction, the fits, the forecast, and the output writer.
- `LICENSE` — the project's licence.

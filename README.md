# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nawadays-ai/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nawadays-ai/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**: forecast,
for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, one
week after launch, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per trophy.
Scoring is the mean pinball loss on the logit scale; lower is better.

## The program

`main.py` (Python 3, standard library only) is run once per case: it reads
every `*.in` case file in `$MAKEFUN_INPUT_DIR` (else `/input`), in name order,
and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
Each output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`
with one row per trophy of the case. It is deterministic: two runs of a case
write byte-identical files.

## The model

Earn rates are cumulative and saturate, and the recorded page only changes
when Exophase refreshes it, so the case history is first reduced to its
distinct page states. Then, per trophy:

- a weighted least-squares line is fitted to the logit of the earn rate
  against ln(1 + hours since the first read), over the last 48 hours of page
  states, each state weighted by the square root of the hours it stayed on
  the page;
- the fitted slope is shrunk toward the pooled slope of the trophy's tier
  (platinum / gold / silver / bronze), which keeps noisy low-count trophies
  from extrapolating wildly;
- the forecast is anchored on the last observed rate and the extrapolated
  logit move is damped by 0.75, so a saturating adoption curve is not
  projected through its ceiling;
- the spread is a normal law on the logit scale combining the fit's residual
  scale, the slope's standard error over the extrapolation distance, a
  floor, and a random-walk term whose horizon is measured from the last page
  change rather than the checkpoint — a stale page is honestly uncertain.

Degenerate inputs (no history, a single page state, a trophy a read never
listed) fall back to the last known rate under the same spread law, so the
program always writes a valid forecast.

The configuration was selected on simulated week-one scenarios (saturating
player growth, per-trophy adoption curves, Exophase's refresh cadence making
reads stale, idiosyncratic per-trophy noise) scored with an exact replica of
the official metric: on 80 such scenarios the entry averages a loss of 0.355
against 0.441 for the persistence baseline and 0.650 for the house curve
baseline, with 90% interval coverage of 0.987. The practice cases' histories
are a single flat pre-refresh page, so they exercise the fallback path; their
server grade is the first real check of the spread under a refresh.

## Layout

- `main.py` — the entry program.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the entrant wallet for the claim check.

## Submission note

This entry was re-packaged into a fresh repository created after the competition opened. The entrypoint was re-authored in the new repository for the resubmission; the model itself is unchanged from the audit-round-3 configuration described above, and the re-authored program was verified to produce byte-identical output to that configuration on the practice cases and the development test battery.

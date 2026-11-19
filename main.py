"""Forecaster for the make.fun bounty gta6-trophy-rarity-forecast-01.

For every PS5 trophy of GTA VI the program predicts the share of
Exophase-tracked players (in percent) that will have earned the trophy at
the case's target moment.  The answer for one case is five quantiles per
trophy; scoring is pinball loss after a logit transform, so all of the
modelling below happens on the logit scale and only the finished quantiles
are mapped back to percent.

Invocation: the grader starts the file once per case with the case files
in an input directory and expects one ``<id>.out`` per ``<id>.in`` in an
output directory.  The directories come from MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR when those variables are set and default to /input and
/output.  Python 3 standard library only, no randomness anywhere: a case
always produces the same bytes.

Model summary (configuration frozen after three external audit rounds;
the constants further down are that frozen configuration):

* Reads of the trophy page repeat between Exophase refreshes, so the
  history is collapsed to the distinct page states it actually shows.
* Each trophy gets a weighted least-squares fit of logit(rate) against
  ln(1 + hours since the first read) on the states of the last 48 hours,
  a state weighing the square root of the time it stayed visible.  Slopes
  are shrunk toward a per-tier pooled slope (the tier median, itself
  shrunk toward the field median), the forecast starts from the last seen
  rate, and the extrapolated move is damped.
* A rate can never drop below last-earners / projected-players, because
  earners only accumulate; the player projection is a damped log fit with
  a floored slope and a capped growth factor.
* Silver/gold trophies sitting near zero may belong to a delayed wave
  (story finale, completion grind) that has not started yet.  A hedge
  handles that case: it only engages when a large share of the silver /
  gold cohort is stalled at once, pulls the centre toward a 2% target
  (a pull, so it cannot overshoot), and widens the upper tail, while a
  reach guard keeps the 5% quantile close to the 0.01% scoring floor.
* The platinum needs every other trophy, so its quantiles are capped,
  level by level, at the rarest other trophy's quantiles whenever the
  last observed page respects that ordering.
* Whatever cannot be fitted falls back to the last known rate under the
  same spread law, and any unexpected failure falls back to a wide
  persistence forecast, so a valid file is always written.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

# The grader launches this file with flags that drop its own directory
# from sys.path; restore it before anything could import a sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- frozen configuration -------------------------------------------------
WINDOW_HOURS = 48.0    # page states older than this leave the trend fit
SHRINK_K = 12.0        # how hard a fitted slope is pulled to its tier pool
TIER_K = 6.0           # how hard a tier pool is pulled to the field pool
MOVE_DAMP = 0.55       # damping applied to the extrapolated logit move
PLAYERS_DAMP = 0.85    # damping applied to the log-players projection
RW_SIGMA = 0.35        # random-walk spread per square-root day (logit)
FLOOR_SIGMA = 0.25     # spread floor (logit)
SPREAD_SCALE = 0.85    # global calibration of the spread
JUMP_CAP = 4.5         # largest allowed |forecast - last| on the logit scale
TAU_MIN = 0.15         # floor for the spread of borrowed slopes
GROWTH_CAP = 5.0       # largest allowed projected growth of the player pool
UP_AT_FLOOR = 1.3      # upper-spread factor when the centre hugs the floor
DOWN_AT_FLOOR = 0.85   # lower-spread factor in the same situation
ASYMMETRY_OFF_PCT = 2.0  # at/above this centre the spread is symmetric
WAVE_TIERS = ("silver", "gold")
WAVE_FULL_PCT = 0.05   # last rate at/below this: fully wave-eligible
WAVE_OFF_PCT = 0.5     # last rate at/above this: not wave-eligible
COHORT_OFF = 0.3       # stalled cohort share at/below this: hedge stays off
COHORT_FULL = 0.7      # stalled cohort share at/above this: hedge fully on
WAVE_HORIZON_DAYS = 1.0  # horizon scaling, days / (days + this)
WAVE_TARGET_PCT = 2.0  # the hedge pulls the centre toward this rate
WAVE_UP = 2.0          # upper-spread factor at full hedge engagement
FLOOR_REACH = 0.75     # how far q05 must reach from the centre to the floor


# --- scalar helpers -------------------------------------------------------

def clamp_rate(rate):
    """Clamp a percent rate to the scorer's range."""
    return min(99.99, max(0.01, rate))


def to_logit(rate_pct):
    p = clamp_rate(rate_pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    # Two branches so exp() never sees a large positive argument.
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round half-up to four decimals; integral results stay integers."""
    scaled = value * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000
    return int(result) if float(result).is_integer() else result


def parse_instant(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


LOGIT_FLOOR = to_logit(0.01)
LOGIT_CEILING = to_logit(99.99)
LOGIT_WAVE_TARGET = to_logit(WAVE_TARGET_PCT)
LOGIT_ASYMMETRY_OFF = to_logit(ASYMMETRY_OFF_PCT)


# --- history handling -----------------------------------------------------

def distinct_states(case):
    """Collapse the read history to distinct page states.

    Returns a list of (hours since the first read, players_tracked, rates).
    A read that repeats the previous kept read exactly is the same page
    state and contributes nothing new, so it is dropped.
    """
    history = case["history"]
    origin = parse_instant(history[0]["at"])
    states = []
    previous_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == previous_key:
            continue
        previous_key = key
        hours = (parse_instant(read["at"]) - origin) / 3600.0
        states.append((hours, read["players_tracked"], list(read["rates"])))
    return states


def newest_values(case):
    """Per trophy, the rate and the player count of its newest sighting.

    Both lists hold None for a trophy no read ever listed.
    """
    count = len(case["trophies"])
    rates = [None] * count
    players = [None] * count
    for read in case["history"]:
        for idx, rate in enumerate(read["rates"]):
            if rate is not None:
                rates[idx] = rate
                players[idx] = read["players_tracked"]
    return rates, players


def state_weights(states, end_hours):
    """Square root of the hours each state stayed on the page."""
    weights = []
    for pos, state in enumerate(states):
        following = states[pos + 1][0] if pos + 1 < len(states) else end_hours
        weights.append(math.sqrt(max(following - state[0], 0.01)))
    return weights


def weighted_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x, taken about weighted means.

    Returns (slope, residual_sd, slope_se, x_mean, y_mean, sxx).  The
    weights are rescaled to mean 1 first: the residual scale divides by
    the weight sum while sxx does not, so without the rescaling the
    slope's standard error would inherit whatever units the raw weights
    happen to carry.
    """
    n = len(xs)
    ws = [w * n / sum(ws) for w in ws]
    total = sum(ws)
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys)]
    if n > 2:
        dof = max(1.0, n - 2)
        resid_sd = math.sqrt(
            sum(w * r * r for r, w in zip(residuals, ws)) / total * (n / dof))
    else:
        resid_sd = 0.0
    slope_se = resid_sd / math.sqrt(sxx) if sxx > 1e-12 else float("inf")
    return slope, resid_sd, slope_se, mean_x, mean_y, sxx


def upper_median(sorted_values):
    """Element at index len//2 of an already sorted list (None if empty)."""
    if not sorted_values:
        return None
    return sorted_values[len(sorted_values) // 2]


# --- the forecast -----------------------------------------------------------

def forecast_case(case):
    levels = case.get("levels") or QUANTILES
    zs = [NormalDist().inv_cdf(level) for level in levels]
    # The reach guard below is calibrated on the 95% point specifically;
    # take its z straight from 0.95 rather than negating the first level's
    # z, which would silently be wrong for a case with different levels.
    z95 = NormalDist().inv_cdf(0.95)
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    def quantile_row(centre, sigma_up, sigma_down):
        return [round4(clamp_rate(from_logit(
            centre + z * (sigma_up if z > 0 else sigma_down)))) for z in zs]

    if not case.get("history"):
        return [{"id": t["id"], "q": quantile_row(to_logit(0.5), 1.5, 1.5)}
                for t in trophies]

    every_state = distinct_states(case)
    last_rate, last_players = newest_values(case)
    first_ts = parse_instant(case["history"][0]["at"])
    checkpoint_h = (parse_instant(case["at"]) - first_ts) / 3600.0
    target_x = math.log(1.0 + checkpoint_h + case["horizon_minutes"] / 60.0)

    # The trend fit only sees the recent window; the oldest state inside
    # it is kept, and the window is ignored if it would leave too little.
    states = every_state
    recent = [s for s in every_state if s[0] >= checkpoint_h - WINDOW_HOURS]
    if len(recent) >= 3:
        states = recent
    weights = state_weights(states, checkpoint_h)

    # Uncertainty grows from the last actual page change, not from the
    # checkpoint: a page that stopped refreshing long ago is stale news.
    stale_days = (checkpoint_h - states[-1][0]) / 24.0
    effective_days = horizon_days + max(0.0, stale_days)
    base_sigma = math.sqrt(FLOOR_SIGMA ** 2
                           + (RW_SIGMA * math.sqrt(effective_days)) ** 2)

    # One trend fit per trophy, on logit(rate) against ln(1 + hours).
    fits = []
    for idx in range(len(trophies)):
        fx, fy, fw = [], [], []
        for pos, (hours, _players, rates) in enumerate(states):
            if rates[idx] is None:
                continue
            fx.append(math.log(1.0 + hours))
            fy.append(to_logit(rates[idx]))
            fw.append(weights[pos])
        if len(fx) >= 3 and len(set(fy)) >= 2:
            fits.append((fx, fy, weighted_line(fx, fy, fw)))
        else:
            fits.append(None)

    def pooled_slope(indices):
        """(Upper median of the fitted slopes, number of fits) for indices.

        A median is used because a single near-collinear fit (two-decimal
        rates can line up almost exactly) reports a near-zero standard
        error and would hijack any precision-weighted pool.
        """
        slopes = sorted(fits[i][2][0] for i in indices if fits[i] is not None)
        return upper_median(slopes), len(slopes)

    field_median, _field_n = pooled_slope(range(len(trophies)))
    field_slope = field_median if field_median is not None else 0.0

    by_tier = {}
    for idx, trophy in enumerate(trophies):
        by_tier.setdefault(trophy["tier"], []).append(idx)
    # Each tier pool is its own median shrunk toward the field median by
    # the number of fits behind it; a tier with no fit at all (the
    # platinum, typically) borrows the field slope outright.
    tier_slope = {}
    for tier, indices in by_tier.items():
        median, n_fits = pooled_slope(indices)
        if median is None:
            tier_slope[tier] = field_slope
        else:
            tier_slope[tier] = ((n_fits * median + TIER_K * field_slope)
                                / (n_fits + TIER_K))

    resid_scales = sorted(fit[2][1] for fit in fits if fit is not None)
    typical_resid = upper_median(resid_scales)
    if typical_resid is None:
        typical_resid = 0.3

    # How much fitted slopes disagree with each other: a trophy that has
    # to borrow its tier's slope inherits this much extra uncertainty.
    borrowed_spread = 0.0
    all_slopes = sorted(fit[2][0] for fit in fits if fit is not None)
    if len(all_slopes) >= 4:
        middle = all_slopes[len(all_slopes) // 2]
        deviations = sorted(abs(s - middle) for s in all_slopes)
        borrowed_spread = max(1.4826 * deviations[len(deviations) // 2],
                              TAU_MIN)

    # Project the tracked-player count to the target with a damped
    # log-players fit over the whole history (the count is smooth, and
    # fitting only the recent window was measured worse: in this data the
    # recent window is the launch surge itself).  The slope cannot go
    # negative and the growth is capped, keeping the projection sane.
    proj_x = [math.log(1.0 + s[0]) for s in every_state]
    proj_y = [math.log(max(s[1], 1)) for s in every_state]
    proj_w = state_weights(every_state, checkpoint_h)
    if len(proj_x) >= 3:
        players_slope = max(weighted_line(proj_x, proj_y, proj_w)[0], 0.0)
        log_target = proj_y[-1] + PLAYERS_DAMP * players_slope * (target_x - proj_x[-1])
        log_target = min(log_target, proj_y[-1] + math.log(GROWTH_CAP))
        players_target = max(math.exp(log_target), every_state[-1][1])
    else:
        players_target = every_state[-1][1]
    last_state_x = math.log(1.0 + states[-1][0])

    # Delayed-wave hedge, cohort gate: the share of silver/gold trophies
    # stalled under the eligibility ceiling.  A whole cohort stalled at
    # once is the signature of a wave that has not started; one stalled
    # trophy among progressing peers is just a hard trophy.
    wave_indices = [i for i, t in enumerate(trophies) if t["tier"] in WAVE_TIERS]
    if wave_indices:
        stalled = sum(1 for i in wave_indices
                      if (last_rate[i] if last_rate[i] is not None else 0.0)
                      < WAVE_OFF_PCT)
        share = stalled / len(wave_indices)
        cohort = min(1.0, max(0.0, (share - COHORT_OFF) / (COHORT_FULL - COHORT_OFF)))
    else:
        cohort = 0.0
    horizon_factor = horizon_days / (horizon_days + WAVE_HORIZON_DAYS)

    rows = []
    for idx, trophy in enumerate(trophies):
        seen_rate = last_rate[idx]
        anchor = to_logit(seen_rate if seen_rate is not None else 0.0)
        fit = fits[idx]
        pool = tier_slope.get(trophy["tier"], field_slope)
        if fit is None:
            # No usable fit of its own: move with the tier pool anyway (a
            # flat line so far says little about the next days, least of
            # all for late-game trophies) and charge the borrowing spread.
            centre = anchor + MOVE_DAMP * pool * (target_x - last_state_x)
            sigma = math.sqrt(base_sigma ** 2 + (borrowed_spread * MOVE_DAMP
                                                 * (target_x - last_state_x)) ** 2)
        else:
            fx, fy, (slope, resid_sd, slope_se, mean_x, _mean_y, sxx) = fit
            weight = len(fx) / (len(fx) + SHRINK_K)
            shrunk = weight * slope + (1.0 - weight) * pool
            centre = fy[-1] + MOVE_DAMP * shrunk * (target_x - fx[-1])
            fit_sigma = resid_sd if resid_sd > 0 else typical_resid
            if resid_sd == 0.0 and sxx > 1e-12:
                # Exactly collinear points carry no residual scale of
                # their own; borrow the pool's scale for the slope error
                # instead of reporting a certain slope.
                slope_se = typical_resid / math.sqrt(sxx)
            sigma = math.sqrt(fit_sigma ** 2
                              + (slope_se * (target_x - mean_x)) ** 2
                              + base_sigma ** 2)

        # Earners never decrease, so the rate has a hard floor: the last
        # earner count spread over the projected player pool.  This is
        # what stops a pure denominator surge from reading as a collapse.
        if seen_rate:
            earners = seen_rate / 100.0 * (last_players[idx] or every_state[-1][1])
            if earners > 0:
                floor_centre = to_logit(earners / players_target * 100.0)
                if centre < floor_centre:
                    centre = floor_centre
        move = centre - anchor
        if move > JUMP_CAP:
            centre = anchor + JUMP_CAP
        elif move < -JUMP_CAP:
            centre = anchor - JUMP_CAP
        centre = min(LOGIT_CEILING, max(LOGIT_FLOOR, centre))

        # Wave hedge: eligibility fades in as the last rate falls to zero
        # and is scaled by the cohort gate and the horizon.  The lift is a
        # pull toward the target logit — it closes part of the remaining
        # gap and can never push the centre past the target.
        effective_rate = seen_rate if seen_rate is not None else 0.0
        engagement = 0.0
        if trophy["tier"] in WAVE_TIERS:
            engagement = (min(1.0, max(0.0, (WAVE_OFF_PCT - effective_rate)
                                       / (WAVE_OFF_PCT - WAVE_FULL_PCT)))
                          * cohort)
        centre = centre + engagement * horizon_factor * max(
            0.0, LOGIT_WAVE_TARGET - centre)
        centre = min(LOGIT_CEILING, max(LOGIT_FLOOR, centre))

        # Near the scoring floor the honest uncertainty points upward:
        # widen the upper spread and narrow the lower one in proportion
        # to how close the centre sits to the floor.
        proximity = min(1.0, max(0.0, (LOGIT_ASYMMETRY_OFF - centre)
                                 / (LOGIT_ASYMMETRY_OFF - LOGIT_FLOOR)))
        sigma_up = sigma * (1.0 + proximity * (UP_AT_FLOOR - 1.0))
        sigma_down = sigma * (1.0 - proximity * (1.0 - DOWN_AT_FLOOR))
        sigma_up *= (1.0 + engagement * horizon_factor * (WAVE_UP - 1.0))
        sigma_up *= SPREAD_SCALE
        sigma_down *= SPREAD_SCALE
        # Reach guard: an engaged trophy's q05 must still reach most of
        # the way down to the floor, so a trophy that truly stays at zero
        # is never left many sigma below the centre.
        needed_down = engagement * FLOOR_REACH * (centre - LOGIT_FLOOR) / z95
        if sigma_down < needed_down:
            sigma_down = needed_down
        rows.append({"id": trophy["id"],
                     "q": quantile_row(centre, sigma_up, sigma_down)})

    # Platinum consistency: the platinum requires every other trophy, so
    # no quantile of it may sit above the same quantile of the rarest
    # other trophy.  The cap is applied level by level (both sequences
    # are non-decreasing, so monotonicity survives) and only while the
    # last observed page itself respects the ordering — a data anomaly
    # disables the cap instead of corrupting the forecast.
    platinum_pos = next((i for i, t in enumerate(trophies)
                         if t["tier"] == "platinum"), None)
    if platinum_pos is not None and last_rate[platinum_pos] is not None:
        other_rates = [last_rate[j] for j in range(len(trophies))
                       if j != platinum_pos]
        if (all(rate is not None for rate in other_rates)
                and last_rate[platinum_pos] <= min(other_rates) + 1e-12):
            plat_row = rows[platinum_pos]["q"]
            for level_pos in range(len(plat_row)):
                cap = min(rows[j]["q"][level_pos]
                          for j in range(len(rows)) if j != platinum_pos)
                if plat_row[level_pos] > cap:
                    plat_row[level_pos] = cap
    return rows


def fallback_rows(case):
    """Wide persistence forecast, used only if the model itself fails:
    a case must never be lost to an exception."""
    levels = case.get("levels") or QUANTILES
    zs = [NormalDist().inv_cdf(level) for level in levels]
    if case.get("history"):
        known, _players = newest_values(case)
    else:
        known = [None] * len(case["trophies"])
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        centre = to_logit(known[idx] if known[idx] is not None else 0.5)
        rows.append({"id": trophy["id"],
                     "q": [round4(from_logit(centre + z * 1.2)) for z in zs]})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            rows = forecast_case(case)
        except Exception:
            rows = fallback_rows(case)
        payload = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()

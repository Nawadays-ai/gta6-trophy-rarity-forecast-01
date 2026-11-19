"""Forecast week-one earn rates for every trophy of a PS5 game.

The input case carries repeated reads of a trophy listing page.  The page
only changes when the publisher refreshes it, so consecutive identical
reads describe a single page state.  This program compresses the reads
into page states, follows each trophy's earn rate on a logit scale as a
function of log elapsed time, and reports five quantiles per trophy for
the target instant.  Only the Python standard library is used.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

# -- tuning constants -------------------------------------------------------
WINDOW_H = 48.0          # hours of page states used by each trophy fit
DAMP = 0.55              # fraction of the fitted move that is extrapolated
SHRINK_K = 12.0          # count shrinkage of a slope toward its tier pool
TIER_K = 6.0             # count shrinkage of a tier pool toward the field
JUMP_CAP = 4.5           # largest logit move away from the anchor
SIG_FLOOR = 0.25         # irreducible logit spread
SIG_RW = 0.35            # random-walk spread per square root day
SIG_MULT = 0.85          # global spread calibration
FALLBACK_SIG = 1.2       # spread used by the emergency rows
EMPTY_SIG = 1.5          # spread used when a case has no history at all
RATE_LO = 0.01           # smallest printable percent
RATE_HI = 99.99          # largest printable percent
HEDGE_TIERS = ("silver", "gold")
HEDGE_TARGET = 2.0       # percent the delayed-wave hedge pulls toward
HEDGE_UP = 2.0           # upper-spread widening strength of the hedge
HEDGE_RAMP_LO = 0.05     # last rate (percent) with full hedge eligibility
HEDGE_RAMP_HI = 0.5      # last rate (percent) where eligibility reaches 0
COHORT_LO = 0.3          # stalled share where the cohort gate opens
COHORT_HI = 0.7          # stalled share where the cohort gate is full
REACH = 0.75             # how far q05 must reach toward the rate floor
TAU_FLOOR = 0.15         # smallest between-trophy slope spread
PLAYERS_GCAP = 5.0       # largest projected growth of the player count

_STD = NormalDist()


def _parse_instant(text):
    """Parse an ISO-8601 timestamp into an aware datetime."""
    if text is None:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def _logit_of_percent(pct):
    """Log-odds of a percent, clamped to the printable range first."""
    p = min(max(float(pct), RATE_LO), RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def _percent_of_logit(value):
    """Inverse of _logit_of_percent without the clamp."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


class PageState:
    """One distinct configuration of the tracked page."""

    __slots__ = ("when", "players", "rates")

    def __init__(self, when, players, rates):
        self.when = when
        self.players = players
        self.rates = rates


def _compress_history(history):
    """Collapse consecutive reads that show the same page into states.

    A read repeats the previous state when both the tracked player count
    and every trophy rate agree.  Each state keeps the moment it was
    first seen; a later state ends the previous one's run, which gives
    the durations used as fit weights.
    """
    states = []
    for read in history or []:
        when = _parse_instant(read.get("at") or read.get("slot"))
        if when is None:
            continue
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if states and states[-1].players == players and states[-1].rates == list(rates):
            continue
        states.append(PageState(when, players, list(rates)))
    return states


def _state_durations(states, end_when):
    """Hours each state was on display; the last one runs to end_when."""
    spans = []
    for idx, state in enumerate(states):
        nxt = states[idx + 1].when if idx + 1 < len(states) else end_when
        if nxt is None:
            nxt = state.when
        spans.append(max(0.01, (nxt - state.when).total_seconds() / 3600.0))
    return spans


class LineFit:
    """Weighted least-squares line plus its uncertainty summary."""

    __slots__ = ("slope", "anchor_x", "resid", "se_slope", "count")

    def __init__(self, slope, anchor_x, resid, se_slope, count):
        self.slope = slope
        self.anchor_x = anchor_x
        self.resid = resid
        self.se_slope = se_slope
        self.count = count


def _fit_line(xs, ys, weights):
    """Fit y ~ a + b*x by weighted least squares.

    Returns None when the points cannot support a line: fewer than
    three points, no spread in x, or no variation in y (a trophy pinned
    at one rate carries no trend of its own).  Weights are rescaled to
    mean one so the slope's standard error does not depend on the
    arbitrary weight scale.
    """
    n = len(xs)
    if n < 3:
        return None
    if min(ys) == max(ys):
        return None
    wsum = sum(weights)
    if wsum <= 0:
        return None
    scale = n / wsum
    ws = [w * scale for w in weights]
    tw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / tw
    my = sum(w * y for w, y in zip(ws, ys)) / tw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    resid = math.sqrt(sum(w * r * r for w, r in zip(ws, residuals)) / (n - 2))
    if resid == 0.0:
        med = _median([abs(r) for r in residuals])
        se_slope = med / math.sqrt(sxx)
    else:
        se_slope = resid / math.sqrt(sxx)
    return LineFit(slope, mx, resid, se_slope, n)


def _tier_pools(fits, trophies):
    """Pool the fitted slopes by tier.

    Each tier contributes the median of its valid slopes; that median is
    then pulled toward the field-wide median in proportion to how many
    valid fits the tier has, so a tier with no fit of its own simply
    borrows the field value.
    """
    by_tier = {}
    every = []
    for idx, fit in fits.items():
        if fit is None:
            continue
        by_tier.setdefault(trophies[idx].get("tier"), []).append(fit.slope)
        every.append(fit.slope)
    field = _median(every) if every else 0.0
    pools = {}
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier")
        if tier in pools:
            continue
        members = by_tier.get(tier)
        if members:
            n = len(members)
            pools[tier] = (n * _median(members) + TIER_K * field) / (n + TIER_K)
        else:
            pools[tier] = field
    return pools, field, every


def _slope_spread(slopes):
    """Robust between-trophy slope spread from the median absolute dev."""
    if not slopes:
        return 0.0
    centre = _median(slopes)
    mad = _median([abs(s - centre) for s in slopes])
    if mad == 0.0:
        return 0.0
    return max(TAU_FLOOR, 1.4826 * mad)


def _project_players(states, xs, spans, x_target, x_last):
    """Project the tracked-player count at the target.

    A log-linear fit over every page state (points weighted by the
    square root of the state durations) gives the growth slope; the
    projection re-anchors on the last observed count, scales the
    growth by 0.85, never lets the slope run negative, and caps total
    growth.  Returns None when there is nothing to project from.
    """
    pts = [(x, math.log(s.players), math.sqrt(span))
           for x, s, span in zip(xs, states, spans)
           if s.players and s.players > 0]
    if not pts:
        return None
    last_log = pts[-1][1]
    slope = 0.0
    if len(pts) >= 2:
        total_w = sum(p[2] for p in pts)
        mx = sum(p[2] * p[0] for p in pts) / total_w
        my = sum(p[2] * p[1] for p in pts) / total_w
        sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
        if sxx > 0:
            slope = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts) / sxx
    slope = max(0.0, slope)
    grown = math.exp(last_log + 0.85 * slope * (x_target - x_last))
    return min(grown, PLAYERS_GCAP * math.exp(last_log))


def _hedge_share(trophies, last_rates):
    """Share of hedge-tier trophies whose last rate sits under the ramp."""
    total = 0
    stalled = 0
    for idx, trophy in enumerate(trophies):
        if trophy.get("tier") in HEDGE_TIERS:
            total += 1
            rate = last_rates[idx]
            if (0.0 if rate is None else rate) < HEDGE_RAMP_HI:
                stalled += 1
    if total == 0:
        return 0.0
    share = stalled / total
    return _clamp((share - COHORT_LO) / (COHORT_HI - COHORT_LO), 0.0, 1.0)


def rows_for_case(case):
    """Build the forecast rows for one case dict."""
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [_STD.inv_cdf(float(lev)) for lev in levels]
    states = _compress_history(case.get("history"))
    if not states:
        return _rows_from_centres(
            [_logit_of_percent(0.5)] * len(trophies), trophies, levels, zs,
            EMPTY_SIG, EMPTY_SIG)
    at_when = _parse_instant(case.get("at")) or states[-1].when
    target_when = _parse_instant(case.get("target")) or at_when
    horizon_min = case.get("horizon_minutes")
    if horizon_min is None:
        horizon_min = (target_when - at_when).total_seconds() / 60.0
    t0 = states[0].when
    xs = [math.log(1.0 + (s.when - t0).total_seconds() / 3600.0) for s in states]
    x_last = xs[-1]
    x_target = math.log(1.0 + (target_when - t0).total_seconds() / 3600.0)
    spans = _state_durations(states, at_when)
    window_start = at_when.timestamp() - WINDOW_H * 3600.0
    in_window = [s for s in states if s.when.timestamp() >= window_start]
    if len(in_window) < 3 and len(states) >= 3:
        window_start = states[-3].when.timestamp()

    fits = {}
    last_rates = [None] * len(trophies)
    last_seen_state = [None] * len(trophies)
    for idx in range(len(trophies)):
        for s_idx in range(len(states) - 1, -1, -1):
            rates = states[s_idx].rates
            if idx < len(rates) and rates[idx] is not None:
                last_rates[idx] = rates[idx]
                last_seen_state[idx] = s_idx
                break
        px, py, pw = [], [], []
        for s_idx, state in enumerate(states):
            if state.when.timestamp() < window_start:
                continue
            rates = state.rates
            if idx < len(rates) and rates[idx] is not None:
                px.append(xs[s_idx])
                py.append(_logit_of_percent(rates[idx]))
                pw.append(math.sqrt(spans[s_idx]))
        fits[idx] = _fit_line(px, py, pw)

    pools, _field, valid_slopes = _tier_pools(
        fits, [t for t in trophies])
    tau = _slope_spread(valid_slopes)
    cohort_f = _hedge_share(trophies, last_rates)
    n_target = _project_players(states, xs, spans, x_target, x_last)
    horizon_days = float(horizon_min) / 1440.0
    h_scale = horizon_days / (horizon_days + 1.0) if horizon_days > 0 else 0.0
    eff_days = max(0.0, (target_when - states[-1].when).total_seconds() / 86400.0)
    rw_term = SIG_RW * math.sqrt(eff_days)
    floor_logit = _logit_of_percent(RATE_LO)
    target_logit = _logit_of_percent(HEDGE_TARGET)
    z95 = _STD.inv_cdf(0.95)

    mus, sig_downs, sig_ups = [], [], []
    for idx, trophy in enumerate(trophies):
        rate = last_rates[idx]
        anchor = _logit_of_percent(rate) if rate is not None else floor_logit
        fit = fits[idx]
        pool = pools.get(trophy.get("tier"), 0.0)
        if fit is not None:
            used = (fit.count * fit.slope + SHRINK_K * pool) / (fit.count + SHRINK_K)
        else:
            used = pool
        move = _clamp(DAMP * used * (x_target - x_last), -JUMP_CAP, JUMP_CAP)
        mu = _clamp(anchor + move, floor_logit, -floor_logit)
        # Earners cannot vanish: a growing player pool dilutes a rate,
        # but the earners already counted stay counted.
        if rate is not None and n_target:
            s_idx = last_seen_state[idx]
            players_then = states[s_idx].players if s_idx is not None else None
            if players_then:
                floor_pct = 100.0 * (rate / 100.0 * players_then) / n_target
                if floor_pct > 0:
                    mu = max(mu, _logit_of_percent(floor_pct))
        var = SIG_FLOOR ** 2 + rw_term ** 2
        if fit is not None:
            var += fit.resid ** 2 + (fit.se_slope * (x_target - fit.anchor_x)) ** 2
        else:
            var += (tau * DAMP * (x_target - x_last)) ** 2
        sig = SIG_MULT * math.sqrt(var)
        # Delayed-wave hedge for silver/gold trophies that have not
        # started: pull the centre toward the target rate and widen
        # the upper side, gated by how much of the cohort is stalled.
        engage = 0.0
        gate = 0.0
        if trophy.get("tier") in HEDGE_TIERS:
            r_last = 0.0 if rate is None else rate
            w_e = _clamp((HEDGE_RAMP_HI - r_last) / (HEDGE_RAMP_HI - HEDGE_RAMP_LO),
                         0.0, 1.0)
            gate = w_e * cohort_f
            engage = gate * h_scale
            if engage > 0 and mu < target_logit:
                mu += engage * (target_logit - mu)
        prox = _clamp((target_logit - mu) / (target_logit - floor_logit), 0.0, 1.0)
        sig_down = sig * (1.0 - 0.15 * prox)
        sig_up = sig * (1.0 + 0.30 * prox)
        if engage > 0:
            sig_up *= 1.0 + 0.5 * HEDGE_UP * engage
            guard = gate * REACH * (mu - floor_logit) / z95
            if guard > sig_down:
                sig_down = guard
        mus.append(mu)
        sig_downs.append(sig_down)
        sig_ups.append(sig_up)

    rows = _rows_from_centres(mus, trophies, levels, zs, sig_downs, sig_ups)
    _apply_platinum_cap(rows, trophies, states)
    return rows


def _round4(value):
    """Round to four decimals; whole numbers are returned as ints."""
    rounded = round(value, 4)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _rows_from_centres(mus, trophies, levels, zs, sig_downs, sig_ups):
    """Turn centres and side spreads into rounded quantile rows."""
    if not isinstance(sig_downs, list):
        sig_downs = [sig_downs] * len(trophies)
        sig_ups = [sig_ups] * len(trophies)
    rows = []
    for idx, trophy in enumerate(trophies):
        qs = []
        for z in zs:
            spread = sig_downs[idx] if z < 0 else sig_ups[idx]
            value = _percent_of_logit(mus[idx] + z * spread)
            value = _clamp(value, RATE_LO, RATE_HI)
            qs.append(_round4(value))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy.get("id"), "q": qs})
    return rows


def _apply_platinum_cap(rows, trophies, states):
    """Cap the platinum row at the rarest other row, level by level.

    Earning the platinum requires every other trophy, so its rate can
    never pass theirs.  The cap is only applied while the last observed
    page itself respects that ordering.
    """
    plat_idx = None
    for idx, trophy in enumerate(trophies):
        if trophy.get("tier") == "platinum":
            plat_idx = idx
            break
    if plat_idx is None or len(rows) < 2:
        return
    last = states[-1].rates if states else []
    if plat_idx >= len(last) or last[plat_idx] is None:
        return
    others = [last[i] for i in range(len(trophies))
              if i != plat_idx and i < len(last) and last[i] is not None]
    if not others or last[plat_idx] > min(others):
        return
    for k in range(len(rows[plat_idx]["q"])):
        cap = min(rows[i]["q"][k] for i in range(len(rows)) if i != plat_idx)
        if rows[plat_idx]["q"][k] > cap:
            rows[plat_idx]["q"][k] = cap


def fallback_rows(case):
    """Emergency rows: last known rate under a plain wide spread.

    Used whenever the main forecast cannot run on a case, so the output
    stays valid no matter how degenerate the input is.  A trophy that
    was never observed is centred on a small default rate.
    """
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [_STD.inv_cdf(float(lev)) for lev in levels]
    history = case.get("history") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                rate = rates[idx]
                break
        centre = _logit_of_percent(rate) if rate is not None else _logit_of_percent(0.5)
        qs = []
        for z in zs:
            value = _percent_of_logit(centre + z * FALLBACK_SIG)
            qs.append(_round4(_clamp(value, RATE_LO, 99.9999)))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy.get("id"), "q": qs})
    return rows


def _process_file(path, out_dir):
    with open(path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    try:
        rows = rows_for_case(case)
    except Exception:
        rows = fallback_rows(case)
    payload = {"trophies": rows}
    name = os.path.splitext(os.path.basename(path))[0] + ".out"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        _process_file(os.path.join(in_dir, name), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

For every trophy in a case, this program forecasts five quantiles of its
earn rate (percent of tracked players who hold it) at the target instant,
from the history of Exophase reads the case carries.  It is a trend
forecaster: launch week is when earn rates move fastest, and a trophy
that sits still today is usually climbing or falling tomorrow.

Model, per trophy
-----------------
Work in logit space, y = logit(p), p the earn rate as a fraction, with half
a phantom player at the extremes only (p is clamped to 0.5/N .. 1 - 0.5/N),
so a trophy nobody holds yet and a trophy everybody holds both stay
finite while interior rates are untouched.
Work in log time, x = ln(1 + t / 12), t the hours since the game's release
(2026-11-19T00:00:00Z): progress through a game is roughly linear in log
time, and log time flattens the future, so a straight line in x is a
saturating curve in t.

For each trophy, fit a weighted least-squares line of y on x over the last
24 hours of reads (every read in that window counts the same).  The slope is
then shrunk toward the median slope of the trophy's tier (platinum, gold,
silver, bronze; tiers with fewer than four trophies borrow the global median
slope instead), because a single trophy's day of reads is noisy -- the
2-decimal rounding alone swings a rare trophy's logit by a lot -- while its
tier-mates share the same launch dynamics.  The shrunk slope is capped at
+/-2.0 so no trophy is ever extrapolated off a cliff.  The forecast anchors
on the fitted level at the last read and moves it by the shrunk slope times
the log-time gap to the target, never more than 4 logit units either way --
except a trophy at 99% or above, which never falls more than 0.5 logit
units (see the v2 changes below).

v2 changes (measured on synthetic final-like cases against the same
scoring, mean pinball loss on the logit scale):
- A rising slope is damped by (1 - p), p the trophy's last rate as a
  fraction.  A straight line in log time overshoots a trophy that is
  already a long way up its S-curve: its true curve flattens as it nears
  saturation, while the line keeps climbing.  The damping costs nothing
  for rare trophies (p near 0) and bites hardest where the overshoot
  lives, at mid rates.  This was the single biggest measured win.
- The slope cap is 2.0, not 2.5: the extra headroom only ever fed
  overshoots on the same flattening trophies.
- A trophy at 99% or above may not fall more than 0.5 logit units.  A
  trophy everybody holds does not lose holders; its rate only ever dips
  when a refresh jumps the player count, and new players earn so trivial
  a trophy within hours, so the rate recovers.  Letting the fitted line
  drag it down several logit units was the worst single-trophy loss in
  testing.
- The spread widening is 1.15, not 1.3: the bands were covering
  everything (90% coverage near 1.00 on synthetic cases), and the
  pinball loss pays for a band that is wider than honest.

v4 changes (measured the same way; v2 0.3422 -> v4 0.2164 mean loss over
18 synthetic cases, tuning 0.3118 -> 0.2020, holdout 0.3802 -> 0.2344):
- Player-surge anticipation.  When the tracked player count jumps by
  more than 1.3x between two consecutive reads inside the last 24 hours,
  the 24h slope fit is contaminated: the surge dilutes every rate, the
  fit sees the drop and the first part of the recovery, and the
  extrapolated median ends up far below where the rate actually lands
  (the new cohort earns its way up the S-curve over the horizon).  So
  when a surge is seen, the forecast median is lifted by
  0.9 * ln(surge ratio) logit units, capped at 0.65.  The lift is proportional to the
  surge, not a fixed constant, so a small refresh step lifts a little
  and a big one lifts a lot.  This was the single biggest v4 win: it
  turns the jump scenarios (v2's worst, up to 0.76 loss) into ordinary
  ones.
- Stronger saturation damping: a rising slope is now damped by
  (1 - p)**5 instead of (1 - p).  Over a 3-5 day horizon the straight
  line in log time overshoots far more than over one day, because the
  S-curve flattens hard as a trophy saturates; the fifth power matches
  that flattening.  Rare trophies (p near 0) are untouched.
- Narrower diffusion, 0.20 instead of 0.35 logit units per root-day:
  once the surge bias is fixed the bands no longer need to cover a
  systematic miss, so the honest random-walk term is smaller.
- A trophy at 99% or above is forecast at 99.99% with a 0.30 spread.
  Near 100% a tenth of a percent is whole logit units, and a trophy
  everybody holds stays everybody's: pinning the median at the ceiling
  removes the worst single-trophy losses.

v5 changes (measured the same way; v4 0.2164 -> v5 0.1916 mean loss over
the same 18 synthetic cases, tuning 0.2020 -> 0.1910, holdout 0.2344 ->
0.1924; wins 14 of 18 scenarios, biggest wins hold2d 0.3800 -> 0.1802,
hold2b 0.3562 -> 0.2389, slowJ60 0.2626 -> 0.2007; only real regression
fastJ48 0.3400 -> 0.3668):
- Horizon-scaled surge shift.  Diagnosing v4 showed the remaining loss
  in jump scenarios is a median bias whose sign and size track the
  horizon: 4.5-day-horizon jump cases under-forecast by +0.19 to +0.67
  logit, while 3-day ones sit near zero.  The recovery from a surge is
  convex -- the new cohort earns its way up the S-curve faster the
  longer it has -- while the fit extrapolates a straight line, so the
  shortfall grows with horizon.  The surge lift is now multiplied by
  (horizon_days / 3) ** 1.25, so a 3-day horizon keeps v4's lift and a
  4.5-day horizon lifts about 1.7x as much.  This was the single biggest
  v5 win, and it wins on tuning and holdout separately.
- Narrower diffusion again, 0.12 instead of 0.20 logit units per
  root-day: with the median bias mostly fixed, the bands need even less
  cover for a systematic miss, so the honest random-walk term shrinks
  further.  Practice cases p00/p01 are unaffected in the median
  (identical medians to v4; no surge fires on their flat history).

Tried and rejected for v5 (each measured on the same 18 cases):
- An excess-ratio surge detector (observed step divided by the
  power-law-expected step, so natural fast growth cannot trip it):
  no better than the plain ratio once the shift is re-tuned.
- A late-jump allowance (shifting the median down in surge cases for
  the second, near-target refresh step): clearly worse, holdout worst
  case nearly doubles -- the visible-jump under-forecast dominates.
- Widening the spread in surge cases, asymmetric spreads, a convexity
  boost for early trophies, horizon-scaled damping, damping exponents
  4 and 6: all worse or flat on one side of the split.

Spread (the five quantiles) is a logit normal around that median with a
standard deviation built from three honest sources: the scatter of the
fit's residuals, the slope's standard error times the extrapolation gap,
and a random-walk term of 0.12 logit units per root-day of horizon for the
moves no trend can see (a refresh landing late, a weekend surge).  The
total is floored at 0.30, capped at 1.30, and widened by 1.15 -- the
pinball loss pays for overconfidence more than for a slightly wide band
when the median itself is uncertain.

Trophies sitting at the rounding floor (last rate 0.01% or below) get
special handling: their history cannot show growth, yet week one is
exactly when mid-game trophies switch on, so their median is lifted to at
least 0.02% and their spread widened to at least 1.3 logit units.  A trophy
that never switches on pays a little for the lift; one that does pays far
less for the cover.

Everything is deterministic: no randomness, no wall-clock reads, no
network.  The same case file always produces the same bytes.

Run as:  python3 main.py
Reads  $MAKEFUN_INPUT_DIR (else /input)  for *.in, writes *.out to
$MAKEFUN_OUTPUT_DIR (else /output).  Python 3.13, standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Log-time scale: progress is measured in ln(1 + hours / LOG_SCALE).
LOG_SCALE = 12.0
# Only reads this recent (hours before the last read) inform the slope.
WINDOW_HOURS = 24.0
# Shrinkage strength of a trophy's slope toward its tier median
# (pseudo-reads).
SHRINK_PRIOR = 4.0
# The shrunk slope never exceeds this, in logit units per log-time unit.
SLOPE_CAP = 2.0
# The forecast never moves more than this many logit units from its anchor.
MOVE_CAP = 4.0
# A trophy at 99% or above never falls more than this many logit units.
TOP_MOVE_CAP = 0.5
# Random-walk diffusion of a rate's logit, per square-root day of horizon.
DIFFUSION = 0.12
# Spread widening applied on top of the combined standard deviation.
SPREAD_SCALE = 1.15
# Saturation damping exponent: a rising slope is multiplied by
# (1 - p) ** DAMP_EXPONENT, p the trophy's last rate as a fraction.
DAMP_EXPONENT = 5.0
# A players surge between two consecutive reads inside the last 24h
# (ratio above this) marks the window as surge-contaminated.
JUMP_RATIO_THRESHOLD = 1.3
# Median lift when a surge is seen, in logit units per ln(surge ratio),
# capped at this many logit units so an extreme refresh step cannot
# throw the median off a cliff.  The lift scales with the horizon:
# (horizon_days / 3) ** JUMP_SHIFT_HORIZON_EXP.
JUMP_SHIFT_K = 0.9
JUMP_SHIFT_CAP = 0.65
JUMP_SHIFT_HORIZON_EXP = 1.25
# A trophy at 99% or above is forecast at this median percent with this
# spread (logit units).
TOP_MEDIAN_PCT = 99.99
TOP_SIGMA = 0.3
# Floor for a trophy at the rounding floor: median at least this percent,
# spread at least this many logit units.
FLOOR_MEDIAN_PCT = 0.02
FLOOR_SIGMA = 1.3
# Fallback for a trophy with no usable history at all.
FALLBACK_MEDIAN_PCT = 0.5
FALLBACK_SIGMA = 1.5
# Release instant of the game (game.release_date in the case, else this).
RELEASE_FALLBACK = "2026-11-19T00:00:00Z"


def parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def logit_of_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def pct_of_logit(y):
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """Round to 4 decimals, half up (the starter's rule); ints stay ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def typical(values):
    """Median of a list (0.0 when empty)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def wls(xs, ys):
    """Ordinary least squares of y on x.  Returns (a, b, se_b, resid_sd).

    a + b*x is the fitted line, se_b the slope's standard error and
    resid_sd the residual standard deviation.  With fewer than two distinct
    x values the slope is zero and the line sits at the mean of y.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    if n > 2:
        s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        s2 = 0.0
    se_b = math.sqrt(s2 / sxx) if s2 > 0.0 else 0.0
    return a, b, se_b, math.sqrt(max(s2, 0.0))


def log_time(hours_since_release):
    """Log-time coordinate, guarded for reads just before release."""
    return math.log(max(1.0 + hours_since_release / LOG_SCALE, 0.01))


def release_instant(case):
    """The game's release instant: game.release_date at midnight UTC."""
    raw = str(case.get("game", {}).get("release_date", "") or "")
    if len(raw) == 10:
        return parse_time(raw + "T00:00:00Z")
    if len(raw) > 10:
        return parse_time(raw)
    return parse_time(RELEASE_FALLBACK)


def fit_one(reads, target_hours):
    """One trophy's own fit.

    reads: list of (hours_since_release, players_tracked, rate_pct_or_None),
    oldest first.  Returns (anchor, slope, se_b, resid_sd, gap, last_rate,
    n_window), or None when there is no usable history.  anchor is the
    fitted logit at the last read; gap is the log-time distance to the
    target; last_rate is the rate in the last read.
    """
    usable = [(t, n, r) for (t, n, r) in reads if r is not None and n and n > 0]
    if not usable:
        return None
    t_last = usable[-1][0]
    x_last = log_time(t_last)
    x_target = log_time(target_hours)
    gap = x_target - x_last
    # Smoothed logit of each read: the rate itself, except at the extremes,
    # where half a phantom player keeps 0% and 100% finite (0.5/N and
    # 1 - 0.5/N).  Interior rates are untouched, so a trophy at 0.63% with
    # four earners still anchors at 0.63%, not at a prior-pulled value.
    points = []
    for (t, n, r) in usable:
        earners = r / 100.0 * n
        p = min(max(earners / n, 0.5 / n), 1.0 - 0.5 / n)
        points.append((t, math.log(p / (1.0 - p))))
    # The slope comes from the recent window only: the launch curve bends,
    # and last week's slope is not this week's.
    window = [(t, y) for (t, y) in points if t >= t_last - WINDOW_HOURS]
    if len(window) < 2:
        window = points[-2:] if len(points) >= 2 else points
    xs = [log_time(t) for (t, _y) in window]
    ys = [y for (_t, y) in window]
    a, b, se_b, resid_sd = wls(xs, ys)
    anchor = a + b * x_last
    return anchor, b, se_b, resid_sd, gap, usable[-1][2], len(window)

def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    release = release_instant(case)
    checkpoint = parse_time(case["at"])
    target_text = case.get("target")
    if target_text:
        target = parse_time(target_text)
    else:
        target = checkpoint
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (target - checkpoint).total_seconds() / 60.0
    horizon_days = max(horizon_minutes, 0) / 1440.0
    target_hours = (target - release).total_seconds() / 3600.0

    # Per-trophy read series, in hours since release.
    series = []
    for index in range(len(trophies)):
        reads = []
        for read in history:
            when = parse_time(read["at"])
            hours = (when - release).total_seconds() / 3600.0
            rates = read.get("rates") or []
            rate = rates[index] if index < len(rates) else None
            players = read.get("players_tracked") or 0
            reads.append((hours, players, rate))
        series.append(reads)

    # Player-surge detection: the largest players jump between two
    # consecutive reads inside the last 24 hours.  A surge dilutes every
    # rate and contaminates the 24h slope fit, so when one is seen the
    # forecast median is lifted below, in proportion to the surge.
    # (players_tracked is per read, the same for every trophy, so the
    # first trophy's series carries the whole story.)
    surge_seen = False
    surge_ratio = 1.0
    if series:
        ref = series[0]
        t_last_all = max((t for (t, _n, _r) in ref), default=0.0)
        for (t0, n0, _r0), (t1, n1, _r1) in zip(ref, ref[1:]):
            if t1 >= t_last_all - WINDOW_HOURS and n0 and n1:
                surge_ratio = max(surge_ratio, n1 / n0)
        surge_seen = surge_ratio > JUMP_RATIO_THRESHOLD

    # First pass: each trophy's own fit, so tier medians can be built.
    fits = [fit_one(reads, target_hours) for reads in series]
    slopes_by_tier = {}
    for trophy, fit in zip(trophies, fits):
        if fit is not None:
            slopes_by_tier.setdefault(trophy.get("tier", "bronze"), []).append(fit[1])
    all_slopes = [fit[1] for fit in fits if fit is not None]
    global_slope = typical(all_slopes)

    rows = []
    for trophy, reads, fit in zip(trophies, series, fits):
        if fit is None:
            rows.append((trophy["id"], FALLBACK_MEDIAN_PCT, FALLBACK_SIGMA))
            continue
        anchor, slope, se_b, resid_sd, gap, last_rate, n_window = fit
        tier = trophy.get("tier", "bronze")
        tier_slopes = slopes_by_tier.get(tier, [])
        prior = typical(tier_slopes) if len(tier_slopes) >= 4 else global_slope
        shrunk = (slope * n_window + prior * SHRINK_PRIOR) / (n_window + SHRINK_PRIOR)
        shrunk = max(min(shrunk, SLOPE_CAP), -SLOPE_CAP)
        if shrunk > 0 and last_rate is not None:
            # A rising line overshoots a trophy that is already far up
            # its S-curve: damp the rise by (1 - p)**5, p the last rate.
            # Over a 3-5 day horizon the flattening is far stronger than
            # over one day, so the fifth power, not the first.
            shrunk *= (1.0 - last_rate / 100.0) ** DAMP_EXPONENT
        y = anchor + shrunk * gap
        if surge_seen:
            # The surge diluted the rates the fit just read; the new
            # cohort earns its way back up the S-curve over the horizon,
            # so the median is lifted in proportion to the surge.  The
            # lift grows with the horizon: the recovery is convex while
            # the fit extrapolates straight, so the shortfall grows with
            # horizon -- a 3-day horizon keeps v4's lift, a 3.75-day
            # horizon lifts about 1.3x as much, a 4.5-day horizon about
            # 1.7x as much.
            horizon_factor = (max(horizon_days, 1.0) / 3.0) ** JUMP_SHIFT_HORIZON_EXP
            y += min(JUMP_SHIFT_K * math.log(surge_ratio), JUMP_SHIFT_CAP) * horizon_factor
        cap = TOP_MOVE_CAP if (last_rate is not None and last_rate >= 99.0) else MOVE_CAP
        y = max(min(y, anchor + cap), anchor - cap)
        median_pct = pct_of_logit(y)
        sigma = math.sqrt(max(resid_sd, 0.15) ** 2
                          + (se_b * gap) ** 2
                          + (DIFFUSION * math.sqrt(max(horizon_days, 0.0))) ** 2)
        sigma = min(max(sigma, 0.30), 1.30) * SPREAD_SCALE
        if last_rate is not None and last_rate >= 99.0:
            # Everybody holds it and everybody keeps holding it: pin
            # the median at the ceiling with a narrow band.  Near 100%
            # a tenth of a percent is whole logit units.
            median_pct = TOP_MEDIAN_PCT
            sigma = TOP_SIGMA
        if last_rate is not None and last_rate <= 0.01:
            # At the rounding floor the history cannot show growth, yet
            # week one is when mid-game trophies switch on: lift the median
            # a touch and widen the band to cover an emergence.
            median_pct = max(median_pct, FLOOR_MEDIAN_PCT)
            sigma = max(sigma, FLOOR_SIGMA)
        rows.append((trophy["id"], median_pct, sigma))

    out_rows = []
    for (tid, median_pct, sigma) in rows:
        center = logit_of_pct(median_pct)
        quantiles = []
        for z in Z_SCORES:
            value = pct_of_logit(center + z * sigma)
            quantiles.append(round4(min(100.0, max(0.0, value))))
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        out_rows.append({"id": tid, "q": quantiles})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

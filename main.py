"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

For every trophy in a case, this program forecasts five quantiles of its earn
rate (percent of tracked players who hold it) at the target instant, from the
history of Exophase reads the case carries.  It is a trend forecaster, not a
persistence one: launch week is when earn rates move fastest, and a trophy
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
+/-2.5 so no trophy is ever extrapolated off a cliff.  The forecast anchors
on the fitted level at the last read and moves it by the shrunk slope times
the log-time gap to the target, never more than 4 logit units either way.

Spread (the five quantiles) is a logit normal around that median with a
standard deviation built from three honest sources: the scatter of the fit's
residuals, the slope's standard error times the extrapolation gap, and a
random-walk term of 0.35 logit units per root-day of horizon for the moves
no trend can see (a refresh landing late, a weekend surge).  The total is
floored at 0.30, capped at 1.30, and widened by 1.3 -- the pinball loss pays
for overconfidence more than for a slightly wide band when the median itself
is uncertain.

Trophies sitting at the rounding floor (last rate 0.01% or below) get special
handling: their history cannot show growth, yet week one is exactly when
mid-game trophies switch on, so their median is lifted to at least 0.02% and
their spread widened to at least 1.3 logit units.  A trophy that never
switches on pays a little for the lift; one that does pays far less for the
cover.

Everything is deterministic: no randomness, no wall-clock reads, no network.
The same case file always produces the same bytes.

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

# Log-time scale: progress is measured in ln(1 + hours / TIME_SCALE).
TIME_SCALE = 12.0
# Only reads this recent (hours before the last read) inform the slope.
WINDOW_HOURS = 24.0
# Shrinkage strength of a trophy's slope toward its tier median (pseudo-reads).
SHRINK_PRIOR = 4.0
# The shrunk slope never exceeds this, in logit units per log-time unit.
SLOPE_CAP = 2.5
# The forecast never moves more than this many logit units from its anchor.
MOVE_CAP = 4.0
# Random-walk diffusion of a rate's logit, per square-root day of horizon.
DIFFUSION = 0.35
# Spread widening applied on top of the combined standard deviation.
SPREAD_SCALE = 1.3
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


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def fit_line(xs, ys):
    """Ordinary least squares of y on x.  Returns (a, b, se_b, resid_sd).

    a + b*x is the fitted line, se_b the slope's standard error and resid_sd
    the residual standard deviation.  With fewer than two distinct x values
    the slope is zero and the line sits at the mean of y.
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
    return math.log(max(1.0 + hours_since_release / TIME_SCALE, 0.01))


def release_instant(case):
    """The game's release instant: game.release_date at midnight UTC."""
    raw = str(case.get("game", {}).get("release_date", "") or "")
    if len(raw) == 10:
        return parse_time(raw + "T00:00:00Z")
    if len(raw) > 10:
        return parse_time(raw)
    return parse_time(RELEASE_FALLBACK)
    """Log-time coordinate, guarded for reads just before release."""
    return math.log(max(1.0 + hours_since_release / TIME_SCALE, 0.01))


def forecast_trophy(reads, target_hours, horizon_days):
    """Median percent and logit sigma for one trophy at the target.

    reads: list of (hours_since_release, players_tracked, rate_pct_or_None),
    oldest first.  target_hours: hours since release at the target.
    """
    usable = [(t, n, r) for (t, n, r) in reads if r is not None and n and n > 0]
    if not usable:
        return FALLBACK_MEDIAN_PCT, FALLBACK_SIGMA
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
    a, b, se_b, resid_sd = fit_line(xs, ys)
    anchor = a + b * x_last
    return anchor, b, se_b, resid_sd, gap, usable[-1][2]


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

    # First pass: each trophy's own fit, so tier medians can be built.
    fits = []
    for reads in series:
        fits.append(forecast_trophy(reads, target_hours, horizon_days))
    slopes_by_tier = {}
    for trophy, fit in zip(trophies, fits):
        if isinstance(fit, tuple) and len(fit) == 6:
            slopes_by_tier.setdefault(trophy.get("tier", "bronze"), []).append(fit[1])
    all_slopes = [fit[1] for fit in fits if isinstance(fit, tuple) and len(fit) == 6]
    global_slope = median_of(all_slopes)

    rows = []
    for trophy, reads, fit in zip(trophies, series, fits):
        if not (isinstance(fit, tuple) and len(fit) == 6):
            median_pct, sigma = fit[0], fit[1]
            rows.append((trophy["id"], median_pct, sigma))
            continue
        anchor, slope, se_b, resid_sd, gap, last_rate = fit
        tier = trophy.get("tier", "bronze")
        tier_slopes = slopes_by_tier.get(tier, [])
        prior = median_of(tier_slopes) if len(tier_slopes) >= 4 else global_slope
        n_window = max(1, sum(1 for (t, _n, _r) in reads
                              if t >= reads[-1][0] - WINDOW_HOURS and _r is not None))
        shrunk = (slope * n_window + prior * SHRINK_PRIOR) / (n_window + SHRINK_PRIOR)
        shrunk = max(min(shrunk, SLOPE_CAP), -SLOPE_CAP)
        y = anchor + shrunk * gap
        y = max(min(y, anchor + MOVE_CAP), anchor - MOVE_CAP)
        median_pct = pct_of_logit(y)
        sigma = math.sqrt(max(resid_sd, 0.15) ** 2
                          + (se_b * gap) ** 2
                          + (DIFFUSION * math.sqrt(max(horizon_days, 0.0))) ** 2)
        sigma = min(max(sigma, 0.30), 1.30) * SPREAD_SCALE
        if last_rate is not None and last_rate <= 0.01:
            # At the rounding floor the history cannot show growth, yet
            # week one is when mid-game trophies switch on: lift the median
            # a touch and widen the band to cover an emergence.
            median_pct = max(median_pct, FLOOR_MEDIAN_PCT)
            sigma = max(sigma, FLOOR_SIGMA)
            y = logit_of_pct(median_pct)
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

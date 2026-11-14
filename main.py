#!/usr/bin/env python3
"""make.fun entry: polymarket-faa-attribution-01 (rebuild).

For each market of a case file, forecast the quantiles of its Yes price
at every minute of the 240 minutes after the checkpoint.

Why this shape
--------------
Polymarket minute prices are sticky: long flat stretches, small ticks,
occasional jumps. The house baseline (last price +/- a fixed normal
spread, sigma 0.0025 per sqrt-minute) ignores what the market's own
recent behaviour says about how wide the next four hours are likely to
be, and about the asymmetry between up and down moves.

So this entry measures the market itself. Over the most recent 8 hours
of its history it collects the empirical distribution of price changes
at a ladder of horizons, reads the requested quantile levels off those
distributions, and interpolates between ladder rungs in sqrt-time, the
scaling a diffusive price follows. The empirical centre (median change)
is shrunk toward zero, because a short window's median drift is mostly
noise while its spread and skew are real signal. A small normal floor
keeps the band from collapsing on a market that has merely been quiet
lately; the market's own measured spread takes over whenever it is
wider.

Standard library only, no network, no randomness: the output is a pure
function of the input file, byte-identical across runs.
"""

import json
import math
import os
from statistics import NormalDist

_GAUSS = NormalDist()

# How much recent history sets the forecast (minutes).
WINDOW_MIN = 480
# Normal floor on the band width, price units per sqrt-minute.
MIN_SIGMA = 0.0005
# Fraction of the measured median change to believe: 0 pins the centre
# on the last price, 1 takes the measured median whole.
CENTRE_KEEP = 0.15
# Horizons (minutes) whose change distributions are measured.
HORIZONS = (1, 2, 4, 8, 15, 30, 60, 120, 180, 240)
# Cap on sampled changes per horizon rung, keeping long histories fast.
SAMPLE_CAP = 1200
# Fallback quantile levels when a case does not name any.
FALLBACK_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)


def clamp01(value):
    """Clamp a float into [0, 1]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def clean_prices(raw):
    """Turn a raw price list into finite floats in [0, 1].

    A bad or missing value repeats the previous good one, so a stray
    hole in the history cannot poison the change distributions. A
    leading hole starts at 0.5.
    """
    cleaned = []
    prev = 0.5
    for value in raw or []:
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = prev
        if not math.isfinite(num):
            num = prev
        cleaned.append(clamp01(num))
        prev = cleaned[-1]
    return cleaned


def sample_quantile(ascending, tau):
    """Linear-interpolated quantile of an ascending sample list."""
    count = len(ascending)
    if count == 0:
        return 0.0
    if count == 1:
        return ascending[0]
    pos = tau * (count - 1)
    lo = int(pos)
    if lo + 1 >= count:
        return ascending[-1]
    frac = pos - lo
    return ascending[lo] * (1.0 - frac) + ascending[lo + 1] * frac


def measure_rungs(prices, taus):
    """Measured change quantiles at each ladder horizon.

    Returns a list of (horizon_minutes, [offset per tau]) with ascending
    horizons, from every `h`-minute change in the window (strided so
    the sample count stays bounded). Horizons at or beyond the window
    length are skipped; callers extrapolate past the last rung.
    """
    n = len(prices)
    measured = []
    for h in HORIZONS:
        if h >= n:
            break
        span = n - h
        stride = max(1, span // SAMPLE_CAP)
        changes = [prices[t + h] - prices[t] for t in range(0, span, stride)]
        changes.sort()
        measured.append((h, [sample_quantile(changes, tau) for tau in taus]))
    return measured


def offsets_for_minute(measured, minute, n_taus):
    """Change quantiles for a given minute, interpolated in sqrt-time."""
    if not measured:
        return [0.0] * n_taus
    first_h, first_offs = measured[0]
    if minute <= first_h:
        scale = math.sqrt(minute / first_h)
        return [off * scale for off in first_offs]
    for i in range(len(measured) - 1):
        h0, offs0 = measured[i]
        h1, offs1 = measured[i + 1]
        if h0 <= minute <= h1:
            w = (math.sqrt(minute) - math.sqrt(h0)) / (
                math.sqrt(h1) - math.sqrt(h0)
            )
            return [a + (b - a) * w for a, b in zip(offs0, offs1)]
    last_h, last_offs = measured[-1]
    scale = math.sqrt(minute / last_h)
    return [off * scale for off in last_offs]


def forecast_one_market(raw_prices, taus, horizon):
    """Quantile rows for one market: `horizon` rows, one per minute."""
    prices = clean_prices(raw_prices)
    if not prices:
        prices = [0.5]
    last_price = prices[-1]
    measured = measure_rungs(prices[-WINDOW_MIN:], taus)
    z_scores = [_GAUSS.inv_cdf(tau) for tau in taus]
    median_pos = taus.index(0.5) if 0.5 in taus else -1
    rows = []
    for minute in range(1, horizon + 1):
        offs = offsets_for_minute(measured, minute, len(taus))
        if median_pos >= 0:
            # Re-centre the row on the shrunk median change, so the band
            # keeps its measured shape around a centre we only partly
            # believe.
            drift = offs[median_pos] * (1.0 - CENTRE_KEEP)
            offs = [off - drift for off in offs]
        root = math.sqrt(minute)
        row = []
        for j, tau in enumerate(taus):
            off = offs[j]
            floor = z_scores[j] * MIN_SIGMA * root
            if tau < 0.5:
                off = min(off, floor)
            elif tau > 0.5:
                off = max(off, floor)
            value = clamp01(last_price + off)
            row.append(round(value, 6))
        row.sort()
        rows.append(row)
    return rows


def run_case(case):
    """Forecast every market of one case dict, defensively."""
    if not isinstance(case, dict):
        return {"forecasts": []}
    try:
        horizon = int(case.get("horizon_minutes") or 240)
    except (TypeError, ValueError):
        horizon = 240
    if horizon <= 0:
        horizon = 240
    try:
        taus = [float(t) for t in (case.get("quantiles") or FALLBACK_TAUS)]
    except (TypeError, ValueError):
        taus = list(FALLBACK_TAUS)
    if not taus:
        taus = list(FALLBACK_TAUS)
    forecasts = []
    for market in case.get("markets") or []:
        if not isinstance(market, dict):
            forecasts.append({"id": None, "q": forecast_one_market([], taus, horizon)})
            continue
        try:
            rows = forecast_one_market(market.get("prices"), taus, horizon)
        except Exception:
            rows = forecast_one_market([], taus, horizon)
        forecasts.append({"id": market.get("id"), "q": rows})
    return {"forecasts": forecasts}


def fallback_case(case):
    """Last-resort valid output: flat 0.5 rows for every named market."""
    if not isinstance(case, dict):
        return {"forecasts": []}
    try:
        horizon = int(case.get("horizon_minutes") or 240)
    except (TypeError, ValueError):
        horizon = 240
    if horizon <= 0:
        horizon = 240
    try:
        taus = [float(t) for t in (case.get("quantiles") or FALLBACK_TAUS)]
    except (TypeError, ValueError):
        taus = list(FALLBACK_TAUS)
    if not taus:
        taus = list(FALLBACK_TAUS)
    forecasts = []
    for market in case.get("markets") or []:
        mid = market.get("id") if isinstance(market, dict) else None
        forecasts.append({"id": mid, "q": [[0.5] * len(taus) for _ in range(horizon)]})
    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    try:
        names = sorted(os.listdir(in_dir))
    except OSError:
        return
    for name in names:
        if not name.endswith(".in"):
            continue
        path = os.path.join(in_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            continue
        try:
            result = run_case(case)
        except Exception:
            result = fallback_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, separators=(",", ":"))
        except OSError:
            pass


if __name__ == "__main__":
    main()

"""Variant B — earners/players dilution forecaster for gta6-trophy-rarity-forecast-01.

Idea: a trophy's earn rate is earners / players. Exophase refreshes irregularly,
so a flat history means "no refresh yet", not "nothing is moving". The model
therefore decomposes every trophy into two counts and extrapolates both in
log-time, then divides:

  players(t):  log P = a + b * ln(1 + hours since first read), fitted by least
               squares when the history actually moves; when it is flat, the
               launch-night prior from the contest's own RULES (638 players at
               the first read, 3,868 at the 04:30Z refresh) gives b = 1.0389.
               Target players are clamped to <= 4x the last read when the
               horizon from the last read exceeds 2 days (launch-week explosion
               is over by then; days 3-7 grow saturating, not 20x).
  earners_i(t): log E_i = c_i + d_i * ln(1 + hours), fitted per trophy when the
               history moves, with slopes shrunk toward their tier mean and the
               tier mean toward the global mean. When the history is flat the
               slope comes from a prior on gamma_i = d logE / d logP:
                 gamma_i = clip(0.07 * (rate0 / 0.63) ** 2.16, 0.02, 1.0)
               (earner growth scales as a power of initial prevalence: the
               trivial 100% trophy tracks players 1:1, early story trophies
               grow fast, late/grind trophies barely move), with one structural
               override: "Complete Chapter N" trophies get gamma 0.80 / 0.35 /
               0.10 for N = 1 / 2 / 3+ (story depth is ordered; chapter 1 fills
               fast, later chapters barely move in week one).
               Earners never decrease: E(target) is floored at the last read.
  rate_i(t) = E_i(t) / P(t), clamped to 0.01..99.99.

Spread: per-tier residual sigma on the logit scale (from the fits, or 0.35
when the history is flat), plus a refresh term (0.30 when the tail of the
history is flat/stale, 0.10 otherwise), plus a horizon term that grows like
0.25 * sqrt(days). Floored at 0.45 so the band never collapses, capped at 1.5.
Quantiles are logit(median) + z * sigma mapped back to percent.

Standard library only. Deterministic: every number is a pure function of the
case file; files are processed in name order; no clock, no randomness.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-night player prior, from the RULES-disclosed first refresh:
# 638 players at the first read, 3,868 at 04:30Z (4.6667 h later).
ANCHOR_B = (math.log(3868.0) - math.log(638.0)) / math.log(1.0 + 4.6667)
ANCHOR_RATE0 = 0.63
PRIOR_C = 0.07
PRIOR_P = 2.16
CHAPTER_GAMMA = {1: 0.80, 2: 0.35}
CHAPTER_GAMMA_LATE = 0.10
GAMMA_LO = 0.02
GAMMA_HI = 1.0
RESID_PRIOR = 0.35
SIGMA_REFRESH_STALE = 0.30
SIGMA_REFRESH_FRESH = 0.10
SIGMA_HORIZON = 0.25
SIGMA_FLOOR = 0.45
SIGMA_CAP = 1.5
PLAYER_CLAMP_DAYS = 2.0
PLAYER_CLAMP_MULT = 4.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, n) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return (a, b, resid, n)


def gamma_prior(trophy, rate_last):
    """Prior earners-growth exponent gamma = d logE / d logP for a flat history."""
    if rate_last is None or rate_last <= 0:
        return 0.05
    if rate_last >= 99.0:
        return 1.0
    desc = trophy.get("description") or ""
    m = re.search(r"Complete Chapter (\d+)", desc)
    if m:
        ch = int(m.group(1))
        if ch in CHAPTER_GAMMA:
            return CHAPTER_GAMMA[ch]
        return CHAPTER_GAMMA_LATE
    g = PRIOR_C * (rate_last / ANCHOR_RATE0) ** PRIOR_P
    return min(GAMMA_HI, max(GAMMA_LO, g))


def tail_is_stale(history):
    """True when the last reads show no refresh: players unchanged and the
    whole rate vector identical over the last 3+ reads."""
    if len(history) < 3:
        return len(history) >= 2 and all(
            h.get("players_tracked") == history[-1].get("players_tracked")
            for h in history
        )
    tail = history[-3:]
    players = {h.get("players_tracked") for h in tail}
    rates = {tuple(h.get("rates") or []) for h in tail}
    return len(players) == 1 and len(rates) == 1


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    t_first = parse_time(history[0]["at"])
    t_last = parse_time(history[-1]["at"])
    t_target = parse_time(case["target"])

    def x_of(t):
        hours = (t - t_first).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    xs_all = [x_of(parse_time(h["at"])) for h in history]
    x_last = xs_all[-1]
    x_target = max(x_last, x_of(t_target))
    days = max(0.0, case["horizon_minutes"] / 1440.0)

    last = history[-1]
    p_last = float(last["players_tracked"])
    rates_last = last["rates"]

    stale = tail_is_stale(history)
    sigma_refresh = SIGMA_REFRESH_STALE if stale else SIGMA_REFRESH_FRESH

    # ---- players law ----
    px, py = [], []
    for h, x in zip(history, xs_all):
        p = h.get("players_tracked")
        if p and p > 0:
            px.append(x)
            py.append(math.log(float(p)))
    # dedup the fit points too: identical (x, logP) pairs add nothing
    seen = set()
    fx, fy = [], []
    for x, y in zip(px, py):
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            fx.append(x)
            fy.append(y)
    player_fit = ols(fx, fy) if len(set(fx)) >= 2 else None
    if player_fit is not None and player_fit[1] > 0.05:
        a_p, b_p = player_fit[0], player_fit[1]
    else:
        a_p, b_p = math.log(p_last), ANCHOR_B
    p_target = math.exp(a_p + b_p * x_target)
    p_target = max(p_target, p_last)
    if days > PLAYER_CLAMP_DAYS:
        p_target = min(p_target, PLAYER_CLAMP_MULT * p_last)

    # ---- earners per trophy ----
    # Usable (x, logE) points per trophy. Consecutive identical reads are
    # collapsed to their first occurrence: a flat run carries no slope
    # information, and keeping every copy would drag fitted slopes to zero.
    keep = []
    for i, h in enumerate(history):
        if keep:
            prev = history[keep[-1]]
            if (h.get("players_tracked") == prev.get("players_tracked")
                    and h.get("rates") == prev.get("rates")):
                continue
        keep.append(i)
    keep_set = set(keep)

    gammas = []
    e_last_list = []
    resid_by_tier = {}
    for i, trophy in enumerate(trophies):
        rate_last = rates_last[i] if i < len(rates_last) else None
        e_last = (rate_last / 100.0 * p_last) if rate_last is not None else 0.0
        e_last_list.append(e_last)
        xs_i, ys_i = [], []
        for j, h in enumerate(history):
            if j not in keep_set:
                continue
            rates = h.get("rates") or []
            p = h.get("players_tracked")
            if i >= len(rates) or rates[i] is None or not p or p <= 0:
                continue
            e = max(rates[i] / 100.0 * float(p), 0.5)
            xs_i.append(xs_all[j])
            ys_i.append(math.log(e))
        fit = ols(xs_i, ys_i) if len(set(xs_i)) >= 2 else None
        if fit is not None and b_p > 0.05:
            g = fit[1] / b_p
            g = min(GAMMA_HI, max(GAMMA_LO, g))
            gammas.append(g)
            tier = trophy.get("tier", "bronze")
            resid_by_tier.setdefault(tier, []).append(fit[2])
        else:
            gammas.append(None)

    # tier-pooled shrinkage: fitted gammas toward their tier mean, tier means
    # toward the global mean; flat-history trophies take the prior.
    fitted = [g for g in gammas if g is not None]
    global_g = sum(fitted) / len(fitted) if fitted else None
    tier_mean = {}
    for tier in {t.get("tier", "bronze") for t in trophies}:
        vals = [g for g, t in zip(gammas, trophies)
                if g is not None and t.get("tier", "bronze") == tier]
        if vals:
            m = sum(vals) / len(vals)
            w = len(vals) / (len(vals) + 5.0)
            tier_mean[tier] = w * m + (1.0 - w) * (global_g if global_g is not None else m)
    for i, trophy in enumerate(trophies):
        if gammas[i] is None:
            gammas[i] = gamma_prior(trophy, rates_last[i] if i < len(rates_last) else None)
        else:
            tier = trophy.get("tier", "bronze")
            if tier in tier_mean:
                w = 0.7
                gammas[i] = w * gammas[i] + (1.0 - w) * tier_mean[tier]
            gammas[i] = min(GAMMA_HI, max(GAMMA_LO, gammas[i]))

    # per-tier residual sigma (fitted) or the flat-history prior
    sigma_tier = {}
    for tier, resids in resid_by_tier.items():
        if resids:
            v = sum(r * r for r in resids) / len(resids)
            sigma_tier[tier] = min(1.0, max(0.25, math.sqrt(v)))
    horizon_term = SIGMA_HORIZON * math.sqrt(days)

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = rates_last[i] if i < len(rates_last) else None
        e_last = e_last_list[i]
        if rate_last is not None and rate_last >= 99.0:
            rate_med = 99.99
        else:
            e_target = e_last * (p_target / p_last) ** gammas[i] if e_last > 0 else 0.0
            e_target = max(e_target, e_last)
            e_target = min(e_target, p_target)
            rate_med = clamp_pct(e_target / p_target * 100.0) if p_target > 0 else 0.01
        tier = trophy.get("tier", "bronze")
        s_resid = sigma_tier.get(tier, RESID_PRIOR)
        sigma = math.sqrt(s_resid ** 2 + sigma_refresh ** 2 + horizon_term ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        lg = logit(rate_med)
        qs = []
        for z in Z:
            v = 100.0 / (1.0 + math.exp(-(lg + z * sigma)))
            qs.append(round4(min(99.99, max(0.01, v))))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

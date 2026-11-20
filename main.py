"""Earn-rate forecaster for gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

For each case the program reads the house's recorded reads of Exophase's GTA VI (PS5)
trophy page and writes five quantiles of every trophy's earn rate at the case's target.
The score is a pinball loss on the logit scale, so everything below works in logit space
and in earners space (rate x players tracked), never in raw percent alone.

How a forecast is made:

  * The page only changes when Exophase refreshes it, so the history is first compressed
    to its distinct page states; repeated reads of one state carry no new information.
  * Players tracked: fitted with a damped power law in hours since release,
    ln P = a + b * ln(t + 2), recency weighted, and read at the target. The pool's growth
    is the dilution force on every rate.
  * Fit mode (two or more distinct states), per trophy: a recency-weighted least-squares
    line of the logit rate against ln(1 + t/24), its slope shrunk toward the mean slope of
    trophies in the same rate band, capped and damped for the extrapolation; and, in
    parallel, an earners model: the trophy's earner count grows with the pool at a
    trophy-specific elasticity (log earners on log players, likewise shrunk), and the
    target rate is earners over the target pool. The median blends the two with
    persistence, the blend trusting the fit more as the horizon grows.
  * Story funnel: chapter trophies and the finale form a sequence. A flow lift keeps a
    later stage's earners from sitting near zero when the fitted trajectory has simply
    not seen the player wave arrive yet: a share of the previous stage's target earners
    (a share that grows with the horizon, since a chapter takes a day or more of play)
    is expected to finish this stage too. Named story missions get a weaker lift off the
    Chapter 2 stock. Forecast medians are then repaired to be non-increasing down the
    funnel, and the platinum's median is capped at the lowest other median.
  * Cold mode (a single distinct state, as in the practice cases, whose page never
    refreshed before the checkpoint): the pool at the target comes from the absolute
    growth law of this game's launch, and each trophy's earners grow at a group
    elasticity measured on that launch (Chapter 1 fastest, deep content near zero),
    under a wide spread, because a stale page is expected to refresh inside the window.
  * Spread: a random-walk term (0.5 per root-day, as the house baseline), the fit's own
    prediction error at the target's leverage, a term for the two models' disagreement,
    and a floor for trophies pinned near 0.01%, whose printed rate is a handful of
    earners and moves in quanta. A trophy every tracked player holds (rate 100) is
    forecast to stay there.

Deterministic: no randomness, no clock, no network; the same case always writes the
same bytes. Runs in milliseconds, far inside the 8 second cap.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Fit-mode constants, calibrated on this game's recorded launch (see README).
DAMP = 0.9            # damping of fitted slopes over the extrapolation
SLOPE_CAP = 2.2       # per unit of ln(1 + t/24)
WALK_SIGMA = 0.5      # random-walk spread per root-day, as the house baseline
FLOOR_WIDEN = 1.3     # extra spread for trophies at the printed floor
HALF_LIFE_H = 12.0    # recency half-life of a page state, hours
FLOW_RHO = 0.35       # share of the previous story stage's earners who finish this one
FLOW_TAU_H = 48.0     # how fast that share is realized with horizon
MISSION_RHO = 0.10    # weaker flow share for named story missions, off Chapter 2

# Cold-mode constants: the absolute pool law of this game's launch and the group
# elasticities of earner growth measured across its first refresh cycle.
P_LAW_A, P_LAW_B = 5.658, 1.365
COLD_SIGMA = 1.7


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def release_epoch(case):
    """Midnight UTC of the game's release date; falls back to the first read's day."""
    try:
        rd = case["game"]["release_date"]
        return parse_ts(rd + "T00:00:00Z" if "T" not in rd else rd)
    except Exception:
        if case["history"]:
            t0 = parse_ts(case["history"][0]["slot"])
            return t0 - (t0 % 86400)
        return parse_ts(case["at"]) - case["horizon_minutes"] * 60


def distinct_states(case, epoch):
    """Compress the history to distinct page states, each with the hours it persisted."""
    states = []
    for h in case["history"]:
        rates = h.get("rates") or []
        key = (h.get("players_tracked"), tuple(-1.0 if r is None else r for r in rates))
        if states and states[-1][0] == key:
            continue
        t = (parse_ts(h["slot"]) - epoch) / 3600.0
        states.append((key, t, h.get("players_tracked"), rates))
    out = []
    for i, (_, t, players, rates) in enumerate(states):
        dur = states[i + 1][1] - t if i + 1 < len(states) else 1.0
        out.append({"t": t, "players": players, "rates": rates, "dur": max(dur, 0.05)})
    return out


def classify(trophy):
    d = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return "platinum"
    if "100% completion" in d:
        return "completion"
    if "complete chapter" in d:
        return "chapter"
    if "complete the final mission" in d:
        return "finale"
    if d.startswith("complete ") and trophy.get("secret") and not any(c.isdigit() for c in d):
        return "mission"
    return "activity"


def chapter_num(trophy):
    digs = "".join(c for c in (trophy.get("description") or "") if c.isdigit())
    return int(digs) if digs else 99


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x; returns (a, b, mean_x, sxx, residual_sd)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, mx, sxx, 1.0
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    dof = max(1.0, sw - 2.0)
    s2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
    return a, b, mx, sxx, math.sqrt(max(s2, 0.0025))


def fit_players(states, t_target):
    """Target pool size: damped power law in ln(t + 2), anchored at the last read."""
    last_p = max(states[-1]["players"] or 1, 1)
    pts = []
    for s in states:
        if not s["players"]:
            continue
        recency = 0.5 ** (max(0.0, states[-1]["t"] - s["t"]) / 36.0)
        pts.append((math.log(max(s["t"], -1.9) + 2), math.log(s["players"]),
                    math.sqrt(s["dur"]) * recency))
    if not pts:
        return float(last_p)
    if len(pts) == 1:
        a, b = pts[0][1] - P_LAW_B * pts[0][0], P_LAW_B
    else:
        sw = sum(w for _, _, w in pts)
        mx = sum(w * x for x, _, w in pts) / sw
        my = sum(w * y for _, y, w in pts) / sw
        sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
        b = sum(w * (x - mx) * (y - my) for x, y, w in pts) / sxx if sxx > 1e-9 else 1.0
        a = my - b * mx
    b = min(max(b, 0.3), 1.8)
    lt_last = math.log(max(states[-1]["t"], -1.9) + 2)
    lt_tgt = math.log(max(t_target, -1.9) + 2)
    ln_law = a + b * 0.9 * lt_tgt + 0.1 * b * lt_last
    ln_anchor = math.log(last_p) + b * 0.85 * (lt_tgt - lt_last)
    p_hat = math.exp(0.5 * ln_law + 0.5 * ln_anchor)
    return min(max(p_hat, last_p), last_p * 40)


# ---------------------------------------------------------------- cold mode

def cold_gamma(trophy, r_last):
    """Earner-growth elasticity by group, measured on this game's launch transition."""
    grp = classify(trophy)
    if grp == "chapter":
        return {1: 1.5, 2: 1.1, 3: 0.35, 4: 0.15}.get(chapter_num(trophy), 0.1)
    if grp == "finale":
        return 0.1
    if grp == "mission":
        return 0.4
    if grp in ("completion", "platinum"):
        return 0.0
    # Activities: the easier, earlier ones grow with the incoming wave; the launch-night
    # rate is the visible part of that, so elasticity rises with it.
    return 0.35 + 0.35 * min(1.0, max(0.0, (logit(r_last) + 4.6) / 1.5))


def cold_forecast(case, states):
    last = states[-1]
    t_target = last["t"] + case["horizon_minutes"] / 60.0
    p_hat = math.exp(P_LAW_A + P_LAW_B * math.log(max(t_target, -1.9) + 2))
    p_hat = max(p_hat, (last["players"] or 1) * 1.05)
    ratio = p_hat / max(last["players"] or 1, 1)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last["rates"][i] if i < len(last["rates"]) else None
        if r is None:
            r = 0.01
        if r >= 99.9:
            rows.append({"id": tr["id"], "q": [99.9, 99.97, 100.0, 100.0, 100.0]})
            continue
        e_hat = max(r / 100 * (last["players"] or 1), 0.35) * ratio ** cold_gamma(tr, r)
        med = logit(max(min(99.0, e_hat / p_hat * 100), 0.01))
        q = sorted(round(inv_logit(med + z * COLD_SIGMA), 4) for z in Z)
        rows.append({"id": tr["id"], "q": q})
    return rows


# ---------------------------------------------------------------- fit mode

def fit_forecast(case, states):
    t_last = states[-1]["t"]
    t_target = t_last + case["horizon_minutes"] / 60.0
    x = lambda t: math.log(1 + max(t, 0.0) / 24.0)
    x_t, x_last = x(t_target), x(t_last)
    h_days = case["horizon_minutes"] / 1440.0
    trophies = case["trophies"]
    n_tr = len(trophies)
    p_hat = fit_players(states, t_target)
    last_p = max(states[-1]["players"] or 1, 1)
    ln_p_ratio = math.log(p_hat / last_p)
    rec = lambda s: 0.5 ** (max(0.0, t_last - s["t"]) / HALF_LIFE_H)

    # Per-trophy fits: logit line in log-time, and earner elasticity on the pool.
    line, elas = [], []
    for i in range(n_tr):
        pts = [(x(s["t"]), logit(s["rates"][i]), math.sqrt(s["dur"]) * rec(s))
               for s in states if i < len(s["rates"]) and s["rates"][i] is not None]
        if len(pts) >= 2:
            a, b, mx, sxx, sd = wls([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        elif pts:
            a, b, mx, sxx, sd = pts[0][1], 0.0, pts[0][0], 0.0, 1.0
        else:
            a, b, mx, sxx, sd = -9.0, 0.0, 0.0, 0.0, 1.0
        line.append({"a": a, "b": b, "mx": mx, "sxx": sxx, "sd": sd, "n": len(pts)})
        ep = [(math.log(max(s["players"] or 1, 1)),
               math.log(max(s["rates"][i] / 100 * (s["players"] or 1), 0.35)),
               math.sqrt(s["dur"]) * rec(s))
              for s in states if i < len(s["rates"]) and s["rates"][i] is not None and s["players"]]
        elas.append(wls([p[0] for p in ep], [p[1] for p in ep], [p[2] for p in ep])[1]
                    if len(ep) >= 2 else 0.6)

    def band(r):
        if r is None or r <= 0.05:
            return "floor"
        if r <= 0.5:
            return "low"
        if r <= 5:
            return "mid"
        return "high"

    band_b, band_g = {}, {}
    for i in range(n_tr):
        r_last_i = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
        bd = band(r_last_i)
        band_b.setdefault(bd, []).append(line[i]["b"])
        band_g.setdefault(bd, []).append(elas[i])
    mean_b = {k: sum(v) / len(v) for k, v in band_b.items()}
    mean_g = {k: sum(v) / len(v) for k, v in band_g.items()}
    all_b = sum(line[i]["b"] for i in range(n_tr)) / n_tr
    all_g = sum(elas) / n_tr

    # How far to trust the fit over persistence grows with the horizon.
    w_blend = min(0.8, 0.18 + 0.6 * (1 - math.exp(-h_days / 2.0)))
    medians, sigmas = [], []
    for i, tr in enumerate(trophies):
        r_last = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
        if r_last is not None and r_last >= 99.9:
            medians.append(9.2)
            sigmas.append(0.12)
            continue
        y_last = logit(r_last) if r_last is not None else -9.0
        f = line[i]
        bd = band(r_last)
        info = f["n"] * max(f["sxx"], 0.0)
        w_s = info / (info + 1.5)
        b_s = w_s * f["b"] + (1 - w_s) * mean_b.get(bd, all_b)
        b_use = max(-SLOPE_CAP, min(SLOPE_CAP, b_s * DAMP))
        med_line = y_last + b_use * (x_t - x_last)
        g_s = max(-0.2, min(2.2, 0.4 * elas[i] + 0.6 * mean_g.get(bd, all_g)))
        e_last = max((r_last or 0.01) / 100 * last_p, 0.35)
        med_e = logit(max(min(99.0, e_last * math.exp(g_s * ln_p_ratio) / p_hat * 100), 0.01))
        # At the printed floor the logit line is quantisation noise; trust earners there.
        med_fit = med_e if (r_last is not None and r_last <= 0.05) else 0.5 * med_line + 0.5 * med_e
        med = w_blend * med_fit + (1 - w_blend) * y_last
        lev = math.sqrt(1.0 + 1.0 / max(f["n"], 1) + (x_t - f["mx"]) ** 2 / max(f["sxx"], 1e-6))
        sd_pred = f["sd"] * min(lev, 3.0)
        walk = WALK_SIGMA * math.sqrt(h_days)
        disagree = abs(med_line - med_e)
        sigma = math.sqrt(walk ** 2 + (0.45 * w_blend * sd_pred) ** 2 + 0.12 ** 2
                          + (0.25 * disagree) ** 2)
        if r_last is not None and r_last <= 0.05:
            sigma *= FLOOR_WIDEN
        medians.append(med)
        sigmas.append(sigma)

    # Funnel flow lift, in earners space (see the module docstring).
    med_rate = [inv_logit(m) for m in medians]
    earn = [r / 100 * p_hat for r in med_rate]
    flow_frac = 1 - math.exp(-(case["horizon_minutes"] / 60.0) / FLOW_TAU_H)

    def lift(idx, anchor_e, rho):
        flow = rho * flow_frac * anchor_e
        if earn[idx] < flow:
            earn[idx] = 0.5 * earn[idx] + 0.5 * flow

    chap = sorted((chapter_num(tr), i) for i, tr in enumerate(trophies) if classify(tr) == "chapter")
    finale = [i for i, tr in enumerate(trophies)
              if classify(tr) == "finale" and "for " not in (tr.get("description") or "").lower()]
    seq = [i for _, i in chap] + finale
    for k in range(1, len(seq)):
        lift(seq[k], earn[seq[k - 1]], FLOW_RHO)
    if len(seq) > 1:
        ch2_e = earn[seq[1]]
        for i, tr in enumerate(trophies):
            if classify(tr) == "mission":
                lift(i, ch2_e, MISSION_RHO)
    med_rate = [min(99.99, e / p_hat * 100) for e in earn]
    medians = [logit(r) for r in med_rate]
    for i, tr in enumerate(trophies):
        r_last = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
        if r_last is not None and r_last >= 99.9:
            medians[i] = 9.2  # the pinned trophy keeps its near-100 median

    # Monotone repair down the story funnel; the platinum sits at the bottom.
    repair = [i for _, i in chap] + [i for i, tr in enumerate(trophies) if classify(tr) == "finale"]
    for k in range(1, len(repair)):
        if med_rate[repair[k]] > med_rate[repair[k - 1]]:
            med_rate[repair[k]] = med_rate[repair[k - 1]]
    plat = [i for i, tr in enumerate(trophies) if classify(tr) == "platinum"]
    if plat:
        p = plat[0]
        med_rate[p] = min(med_rate[p], min(med_rate[j] for j in range(n_tr) if j != p))
    medians = [logit(r) for r in med_rate]
    for i, tr in enumerate(trophies):
        r_last = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
        if r_last is not None and r_last >= 99.9:
            medians[i] = 9.2

    rows = []
    for i, tr in enumerate(trophies):
        q = sorted(round(inv_logit(medians[i] + z * sigmas[i]), 4) for z in Z)
        rows.append({"id": tr["id"], "q": q})
    return rows


def forecast_case(case):
    epoch = release_epoch(case)
    states = distinct_states(case, epoch)
    if not states:
        raise ValueError("case has no history")
    if len(states) <= 1:
        return cold_forecast(case, states)
    return fit_forecast(case, states)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        rows = forecast_case(case)
        out = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

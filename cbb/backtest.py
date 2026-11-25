"""Walk-forward backtest: for each day, fit only on games before that day, price that day's
games against ESPN's lines, and grade against the final score. No look-ahead."""
import datetime as dt
import itertools
import json
import math

import numpy as np
import pandas as pd

from . import model, store

BUCKETS = [(0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 9.0)]


def walk_forward(games, p, seasons, step=1, skip_days=0):
    done = store.finished(games).copy()
    pool = model.pooled_ids(done, p.pool_min)
    done["et_day"] = store.to_et_date(done["date"])
    out = []
    for season in seasons:
        sg = done[done["season"] == season]
        days = sorted(sg["et_day"].unique())
        if not days:
            continue
        start = days[0] + dt.timedelta(days=skip_days)
        days = [d for d in days if d >= start]
        ratings = None
        for i, day in enumerate(days):
            if ratings is None or i % step == 0:
                as_of = pd.Timestamp(dt.datetime.combine(day, dt.time(5, 0)), tz="UTC")  # ~midnight ET
                prior = done[pd.to_datetime(done["date"], utc=True) < as_of]
                ratings = model.fit(prior, as_of, season, p, pool)
            if ratings is None:
                continue
            for g in sg[sg["et_day"] == day].to_dict("records"):
                mg, tot, _, _ = ratings.predict(g["home_id"], g["away_id"], g["neutral"] == 1, pool)
                out.append({**g, "pred_margin": mg, "pred_total": tot})
    return pd.DataFrame(out)


def _bucket_table(rows):
    tab = []
    for lo, hi in BUCKETS:
        b = [r for r in rows if lo <= r["ev"] < hi]
        if not b:
            tab.append({"ev": f"{lo:.0%}+" if hi > 1 else f"{lo:.0%}-{hi:.0%}", "n": 0})
            continue
        prof = [r["profit"] for r in b]
        wins = sum(1 for x in prof if x > 0)
        losses = sum(1 for x in prof if x < 0)
        tab.append({
            "ev": f"{lo:.0%}+" if hi > 1 else f"{lo:.0%}-{hi:.0%}",
            "n": len(b), "w": wins, "l": losses, "p": len(b) - wins - losses,
            "win_pct": wins / max(1, wins + losses),
            "roi": sum(prof) / len(b),
            "units": sum(prof),
            # one standard error of ROI, so a 'profitable' bucket can be judged against noise
            "roi_se": float(np.std(prof) / math.sqrt(len(b))) if len(b) > 1 else None,
        })
    return tab


def evaluate(pred, p):
    if pred.empty:
        return {}
    res = {"games": int(len(pred))}
    m_err = (pred["home_score"] - pred["away_score"]) - pred["pred_margin"]
    t_err = (pred["home_score"] + pred["away_score"]) - pred["pred_total"]
    res["model_margin_mae"] = float(m_err.abs().mean())
    res["model_total_mae"] = float(t_err.abs().mean())
    res["margin_sd"] = float(m_err.std())
    res["total_sd"] = float(t_err.std())
    ls = pred[pred["spread"].notna()]
    if len(ls):
        res["n_spread_lines"] = int(len(ls))
        res["line_margin_mae"] = float(((ls["home_score"] - ls["away_score"]) + ls["spread"]).abs().mean())
        res["model_margin_mae_same_games"] = float(
            ((ls["home_score"] - ls["away_score"]) - ls["pred_margin"]).abs().mean())
    lt = pred[pred["total"].notna()]
    if len(lt):
        res["n_total_lines"] = int(len(lt))
        res["line_total_mae"] = float(((lt["home_score"] + lt["away_score"]) - lt["total"]).abs().mean())
        res["model_total_mae_same_games"] = float(
            ((lt["home_score"] + lt["away_score"]) - lt["pred_total"]).abs().mean())

    bets = {"spread": [], "total": [], "ml": []}
    for g in pred.to_dict("records"):
        priced = model.price_game(g["pred_margin"], g["pred_total"], g, p)
        for mk, b in priced.items():
            prof = model.grade(b, mk, g["home_score"], g["away_score"])
            bets[mk].append({"ev": b["ev"], "profit": prof})
    res["buckets"] = {mk: _bucket_table(rows) for mk, rows in bets.items()}
    return res


def tune(games, seasons_fit, step=3):
    """Grid-search the rating parameters on walk-forward accuracy (mean absolute error of
    margin and total vs actual results) over the given seasons. Betting results are NOT the
    objective, which keeps the search from fitting to noise in win/loss records."""
    grid = {"half_life": [30, 60, 120], "lam": [2, 5, 12, 30], "prev_w": [0.15, 0.35, 0.7]}
    best, best_score, log = None, float("inf"), []
    for hl, lam, pw in itertools.product(*grid.values()):
        p = model.Params(half_life=hl, lam=lam, prev_w=pw)
        pred = walk_forward(games, p, seasons_fit, step=step)
        if pred.empty:
            continue
        mae_m = ((pred["home_score"] - pred["away_score"]) - pred["pred_margin"]).abs().mean()
        mae_t = ((pred["home_score"] + pred["away_score"]) - pred["pred_total"]).abs().mean()
        score = mae_m + mae_t
        log.append({"half_life": hl, "lam": lam, "prev_w": pw, "mae_margin": mae_m, "mae_total": mae_t})
        if score < best_score:
            best, best_score = p, score
    return best, log


def fit_blend(pred, p):
    """Choose how much weight the model gets against the market line (minimum absolute error
    on games with lines), and measure the error spread of that blend for the probabilities."""
    out = {}
    for col, line, actual, sign in (("pred_margin", "spread", "margin", -1), ("pred_total", "total", "total", 1)):
        g = pred[pred[line].notna()]
        if len(g) < 300:
            continue
        y = (g["home_score"] - g["away_score"]) if actual == "margin" else (g["home_score"] + g["away_score"])
        mkt = sign * g[line]
        best = min((((w * g[col] + (1 - w) * mkt) - y).abs().mean(), w) for w in np.arange(0, 1.001, 0.05))
        w = float(round(best[1], 2))
        sd = float(((w * g[col] + (1 - w) * mkt) - y).std())
        out[actual] = (w, sd, float(best[0]), float((mkt - y).abs().mean()))
    return out


def run_backtest(games, p, cur_season):
    """Report on last season (used for tuning, so in-sample for the 3 knobs) and on the current
    season (out of sample)."""
    report = {"params": p.to_dict(), "generated": store.now_et().isoformat()}
    for label, seasons in (("last_season", [cur_season - 1]), ("this_season", [cur_season])):
        pred = walk_forward(games, p, seasons, step=1)
        report[label] = evaluate(pred, p)
    store.BACKTEST.write_text(json.dumps(report, indent=1, default=float))
    return report

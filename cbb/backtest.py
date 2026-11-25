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


def walk_forward(games, p, seasons, step=1, skip_days=0, which=("pts", "eff")):
    """Per-game out-of-sample forecasts from each model plus the tuned mix."""
    done = store.finished(games).copy()
    pool = model.pooled_ids(done, p.pool_min)
    done["et_day"] = store.to_et_date(done["date"])
    dts = pd.to_datetime(done["date"], utc=True)
    out = []
    for season in seasons:
        sg = done[done["season"] == season]
        days = sorted(sg["et_day"].unique())
        if not days:
            continue
        start = days[0] + dt.timedelta(days=skip_days)
        days = [d for d in days if d >= start]
        rp = re = None
        for i, day in enumerate(days):
            if rp is None or i % step == 0:
                as_of = pd.Timestamp(dt.datetime.combine(day, dt.time(5, 0)), tz="UTC")  # ~midnight ET
                prior = done[dts < as_of]
                rp = model.fit(prior, as_of, season, p, pool) if "pts" in which else None
                re = model.fit_eff(prior, as_of, season, p, pool) if "eff" in which else None
            for g in sg[sg["et_day"] == day].to_dict("records"):
                row = dict(g)
                if rp is not None:
                    row["pm_pts"], row["pt_pts"], _, _ = rp.predict(g["home_id"], g["away_id"], g["neutral"] == 1, pool)
                if re is not None:
                    row["pm_eff"], row["pt_eff"], _, _ = re.predict(g["home_id"], g["away_id"], g["neutral"] == 1, pool)
                if "pm_pts" in row or "pm_eff" in row:
                    out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for k in ("pm_pts", "pt_pts", "pm_eff", "pt_eff"):
        if k not in df.columns:
            df[k] = np.nan
    # efficiency forecast missing (too few box scores yet) -> fall back to points model
    df["pm_eff"] = df["pm_eff"].fillna(df["pm_pts"])
    df["pt_eff"] = df["pt_eff"].fillna(df["pt_pts"])
    df["pm_pts"] = df["pm_pts"].fillna(df["pm_eff"])
    df["pt_pts"] = df["pt_pts"].fillna(df["pt_eff"])
    df["pred_margin"] = (1 - p.mix_margin) * df["pm_pts"] + p.mix_margin * df["pm_eff"]
    df["pred_total"] = (1 - p.mix_total) * df["pt_pts"] + p.mix_total * df["pt_eff"]
    return df


def _bucket_table(rows):
    tab = []
    for lo, hi in BUCKETS:
        b = [r for r in rows if lo <= r["ev"] < hi]
        if not b:
            tab.append({"ev": f"{lo:.0%}+" if hi > 1 else f"{lo:.0%}-{hi:.0%}", "n": 0})
            continue
        prof = [r["profit"] for r in b]
        risk = [r["risk"] for r in b]
        wins = sum(1 for x in prof if x > 0)
        losses = sum(1 for x in prof if x < 0)
        tab.append({
            "ev": f"{lo:.0%}+" if hi > 1 else f"{lo:.0%}-{hi:.0%}",
            "n": len(b), "w": wins, "l": losses, "p": len(b) - wins - losses,
            "win_pct": wins / max(1, wins + losses),
            "roi": sum(prof) / sum(risk),          # return on units risked
            "units": sum(prof),
            "risked": sum(risk),
            # one standard error of ROI, so a 'profitable' bucket can be judged against noise
            "roi_se": float(np.std(prof) / math.sqrt(len(b)) / np.mean(risk)) if len(b) > 1 else None,
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
        for k in ("pts", "eff"):
            res[f"{k}_margin_mae_same_games"] = float(((ls["home_score"] - ls["away_score"]) - ls[f"pm_{k}"]).abs().mean())
    lt = pred[pred["total"].notna()]
    if len(lt):
        res["n_total_lines"] = int(len(lt))
        res["line_total_mae"] = float(((lt["home_score"] + lt["away_score"]) - lt["total"]).abs().mean())
        res["model_total_mae_same_games"] = float(
            ((lt["home_score"] + lt["away_score"]) - lt["pred_total"]).abs().mean())
        for k in ("pts", "eff"):
            res[f"{k}_total_mae_same_games"] = float(((lt["home_score"] + lt["away_score"]) - lt[f"pt_{k}"]).abs().mean())

    bets = {"spread": [], "total": [], "ml": []}
    for g in pred.to_dict("records"):
        priced = model.price_game(g["pred_margin"], g["pred_total"], g, p)
        month = pd.Timestamp(g["date"]).tz_convert(store.ET).strftime("%b")
        sz = abs(g["spread"]) if g.get("spread") == g.get("spread") and g.get("spread") is not None else None
        size = None if sz is None else "0-4.5" if sz <= 4.5 else "5-9.5" if sz <= 9.5 else "10-15.5" if sz <= 15.5 else "16+"
        for mk, b in priced.items():
            prof = model.grade(b, mk, g["home_score"], g["away_score"])
            bets[mk].append({"ev": b["ev"], "profit": prof, "risk": model.stake(b["odds"]), "month": month, "size": size,
                             "conf": "Conference" if g.get("conf_game") == 1 else "Non-conf"})
    res["buckets"] = {mk: _bucket_table(rows) for mk, rows in bets.items()}
    # flagged plays split by game type, to look for pockets where the edge is real.
    # Many splits = some will look good by chance; trust only ones that hold up for months.
    seg = {}
    for mk, rows in bets.items():
        plays = [r for r in rows if r["ev"] >= p.min_ev]
        seg[mk] = {}
        for key in ("month", "conf", "size"):
            groups = {}
            for r in plays:
                if r[key] is not None:
                    groups.setdefault(r[key], []).append((r["profit"], r["risk"]))
            seg[mk][key] = [{"group": k, "n": len(v), "units": sum(x for x, _ in v),
                             "roi": sum(x for x, _ in v) / sum(y for _, y in v),
                             "roi_se": float(np.std([x for x, _ in v]) / math.sqrt(len(v)) / np.mean([y for _, y in v]))
                             if len(v) > 1 else None}
                            for k, v in groups.items()]
    res["segments"] = seg
    return res


def _mae_pred(pred, mcol, tcol):
    m = ((pred["home_score"] - pred["away_score"]) - pred[mcol]).abs().mean()
    t = ((pred["home_score"] + pred["away_score"]) - pred[tcol]).abs().mean()
    return m, t


def tune(games, seasons_fit, step=3):
    """Grid-search each model's knobs on walk-forward accuracy (mean absolute error of margin
    plus total vs actual results). Betting results are NOT the objective, which keeps the
    search from fitting to noise in win/loss records."""
    grid = {"half_life": [30, 60, 120], "lam": [2, 5, 12, 30], "prev_w": [0.15, 0.35, 0.7]}
    best, log = model.Params(), []
    for kind, pre in (("pts", ""), ("eff", "e_")):
        best_score = float("inf")
        lam_grid = grid["lam"] if kind == "pts" else [3, 8, 20, 50]
        for hl, lam, pw in itertools.product(grid["half_life"], lam_grid, grid["prev_w"]):
            p = model.Params(**{**best.to_dict(), pre + "half_life": hl, pre + "lam": lam, pre + "prev_w": pw})
            pred = walk_forward(games, p, seasons_fit, step=step, which=(kind,))
            if pred.empty or pred[f"pm_{kind}"].isna().all():
                continue
            mm, mt = _mae_pred(pred, f"pm_{kind}", f"pt_{kind}")
            log.append({"model": kind, "half_life": hl, "lam": lam, "prev_w": pw, "mae_margin": mm, "mae_total": mt})
            if mm + mt < best_score:
                best_score, chosen = mm + mt, p
        if best_score < float("inf"):
            best = chosen
    return best, log


def fit_mix(pred):
    """Jointly choose (a) the efficiency model's share of the ratings forecast and (b) the
    ratings' weight against the market line, by minimum absolute error on games with lines.
    Also returns the error spread of that final forecast, used for the probabilities."""
    out = {}
    specs = (("margin", "spread", -1, "pm_pts", "pm_eff"), ("total", "total", 1, "pt_pts", "pt_eff"))
    for name, line, sign, c1, c2 in specs:
        g = pred[pred[line].notna()]
        if len(g) < 200:
            continue
        y = (g["home_score"] - g["away_score"]) if name == "margin" else (g["home_score"] + g["away_score"])
        mkt = (sign * g[line]).to_numpy()
        a, b, y = g[c1].to_numpy(), g[c2].to_numpy(), y.to_numpy()
        best = None
        for s_ in np.arange(0, 1.001, 0.1):
            r = (1 - s_) * a + s_ * b
            for w in np.arange(0, 1.001, 0.05):
                mae = np.abs(w * r + (1 - w) * mkt - y).mean()
                if best is None or mae < best[0]:
                    best = (mae, round(float(s_), 2), round(float(w), 2))
        mae, s_, w = best
        f = w * ((1 - s_) * a + s_ * b) + (1 - w) * mkt
        out[name] = {"mix": s_, "w": w, "sd": float((f - y).std()), "mae": float(mae),
                     "market_mae": float(np.abs(mkt - y).mean()),
                     "pts_mae": float(np.abs(a - y).mean()), "eff_mae": float(np.abs(b - y).mean()), "n": int(len(g))}
    return out


def run_backtest(games, p, cur_season):
    """Report on last season (used for tuning, so in-sample for the 3 knobs) and on the current
    season (out of sample)."""
    report = {"params": p.to_dict(), "generated": store.now_et().isoformat(),
              "mix_fit": json.loads(store.PARAMS.read_text()).get("mix_fit") if store.PARAMS.exists() else None}
    for label, seasons in (("last_season", [cur_season - 1]), ("this_season", [cur_season])):
        pred = walk_forward(games, p, seasons, step=1)
        report[label] = evaluate(pred, p)
    store.BACKTEST.write_text(json.dumps(report, indent=1, default=float))
    return report

"""Entry point.  python -m cbb.run update | tune | backtest"""
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

from . import backtest, espn, model, site, store

FETCHED = store.DATA / "fetched_days.txt"


def _fetched():
    return set(FETCHED.read_text().split()) if FETCHED.exists() else set()


def pull(days, games, force=()):
    done = _fetched()
    rows, n = [], 0
    for d in days:
        key = d.strftime("%Y%m%d")
        if key in done and key not in force:
            continue
        try:
            rows += espn.parse_day(espn.scoreboard(key))
        except Exception as e:  # leave the day unfetched; the next run retries it
            print(f"  ! {key}: {e}", flush=True)
            continue
        if d <= store.now_et().date() - dt.timedelta(days=2):
            done.add(key)  # only lock in days whose games are all final
        n += 1
        if n % 25 == 0:  # checkpoint long backfills
            games = store.upsert_games(games, rows)
            rows = []
            FETCHED.write_text("\n".join(sorted(done)))
            print(f"  ...{n} days fetched", flush=True)
        if not os.environ.get("CBB_FAKE"):
            time.sleep(0.4)  # ESPN rate-limits bursts
    games = store.upsert_games(games, rows)
    store.DATA.mkdir(exist_ok=True)
    FETCHED.write_text("\n".join(sorted(done)))
    return games


LINES_CHECKED = store.DATA / "lines_checked.txt"


def backfill_lines(games, cur, budget):
    """Fill closing lines for finished games from ESPN game summaries, a batch per run,
    last season first (it is what tuning and the backtest need)."""
    checked = set(LINES_CHECKED.read_text().split()) if LINES_CHECKED.exists() else set()
    fin = store.finished(games)
    todo = fin[fin["spread"].isna() & fin["total"].isna() & (fin["season"] >= cur - 1)
               & ~fin["id"].isin(checked)]
    todo = todo.assign(pri=(todo["season"] != cur - 1).astype(int)).sort_values(["pri", "date"])
    todo = todo.head(budget)
    if todo.empty:
        return games, 0
    print(f"Backfilling closing lines for {len(todo)} games ({len(fin[fin['spread'].isna()])} missing)...", flush=True)
    idx = games.set_index("id")
    found = 0
    for i, g in enumerate(todo.to_dict("records")):
        try:
            o = espn.summary_odds(g["id"], g["home_abbr"], g["away_abbr"])
        except Exception as e:
            print(f"  ! summary {g['id']}: {e}", flush=True)
            continue
        checked.add(g["id"])
        if o:
            for k, v in o.items():
                if v is not None:
                    idx.at[g["id"], k] = v
            found += 1
        if not os.environ.get("CBB_FAKE"):
            time.sleep(0.25)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1} done, {found} with lines", flush=True)
    games = idx.reset_index()
    games.to_csv(store.GAMES, index=False)
    LINES_CHECKED.write_text("\n".join(sorted(checked)))
    print(f"  lines found for {found}/{len(todo)}", flush=True)
    return games, found


def load_params():
    if store.PARAMS.exists():
        return model.Params.from_dict(json.loads(store.PARAMS.read_text()))
    return model.Params()


def do_tune(games, cur):
    print("Tuning rating parameters on last season (walk-forward)...", flush=True)
    best, log = backtest.tune(games, [cur - 1])
    if best is None:
        best = model.Params()
    pred = backtest.walk_forward(games, best, [cur - 1], step=1)
    if not pred.empty:  # probabilities use the real error spread, measured walk-forward
        best.margin_sd = round(float(((pred["home_score"] - pred["away_score"]) - pred["pred_margin"]).std()), 2)
        best.total_sd = round(float(((pred["home_score"] + pred["away_score"]) - pred["pred_total"]).std()), 2)
        b = backtest.fit_blend(pred, best)
        if "margin" in b:
            best.w_margin, best.margin_sd = b["margin"][0], round(b["margin"][1], 2)
        if "total" in b:
            best.w_total, best.total_sd = b["total"][0], round(b["total"][1], 2)
        print("Blend (weight, sd, blend MAE, market MAE):", b, flush=True)
    store.PARAMS.write_text(json.dumps({**best.to_dict(), "blend_fit": "margin" in b if not pred.empty else False,
                                        "tune_log": log}, indent=1, default=float))
    print("Chosen:", best.to_dict(), flush=True)
    return best


def update_picks(games, ratings, pool, p, now_utc):
    picks = json.loads(store.PICKS.read_text()) if store.PICKS.exists() else {}
    t = pd.to_datetime(games["date"], utc=True)
    upcoming = games[(games["state"] == "pre") & (t > now_utc) & (t < now_utc + pd.Timedelta(hours=40))]
    slate = []
    for g in upcoming.to_dict("records"):
        mg, tot, hp, ap = ratings.predict(g["home_id"], g["away_id"], g["neutral"] == 1, pool)
        priced = model.price_game(mg, tot, g, p)
        snap = {"time": now_utc.isoformat(), "pred_margin": mg, "pred_total": tot, "bets": priced,
                "spread": g.get("spread"), "total": g.get("total"),
                "ml_home": g.get("ml_home"), "ml_away": g.get("ml_away")}
        rec = picks.get(g["id"])
        if rec is None and priced:
            rec = picks[g["id"]] = {"id": g["id"], "date": g["date"], "home": g["home"], "away": g["away"],
                                    "first": snap}
        if rec is not None:
            rec["last"] = snap
        slate.append({**{k: g.get(k) for k in ("id", "date", "home", "away", "neutral", "spread", "total",
                                               "ml_home", "ml_away", "book", "spread_open", "total_open")},
                      "pred_home": hp, "pred_away": ap, "pred_margin": mg, "pred_total": tot,
                      "blend_margin": model.blend(mg, tot, g, p)[0], "blend_total": model.blend(mg, tot, g, p)[1],
                      "home_games": ratings.n_games.get(g["home_id"], 0),
                      "away_games": ratings.n_games.get(g["away_id"], 0),
                      "bets": priced})
    # attach final scores
    fin = store.finished(games).set_index("id")
    for gid, rec in picks.items():
        if gid in fin.index and "final" not in rec:
            rec["final"] = [float(fin.at[gid, "home_score"]), float(fin.at[gid, "away_score"])]
    store.PICKS.write_text(json.dumps(picks, indent=0, default=float))
    return picks, sorted(slate, key=lambda s: s["date"])


def record(picks, p):
    """Live results of flagged plays. 'first' = the price when the model first saw the game
    (what you could have bet); CLV compares it to the last line before tip."""
    out = {mk: {"n": 0, "w": 0, "l": 0, "push": 0, "units": 0.0, "clv_sum": 0.0, "clv_n": 0, "clv_pos": 0}
           for mk in ("spread", "total", "ml")}
    for rec in picks.values():
        if "final" not in rec:
            continue
        hs, as_ = rec["final"]
        for mk in out:
            b = rec["first"]["bets"].get(mk)
            if not b or b["ev"] < p.min_ev:
                continue
            o = out[mk]
            prof = model.grade(b, mk, hs, as_)
            o["n"] += 1
            o["units"] += prof
            o["w" if prof > 0 else "l" if prof < 0 else "push"] += 1
            last = rec.get("last", rec["first"])
            clv = None
            if mk == "spread" and last.get("spread") is not None:
                close = last["spread"] if b["side"] == "home" else -last["spread"]
                clv = b["line"] - close
            elif mk == "total" and last.get("total") is not None:
                clv = (last["total"] - b["line"]) * (1 if b["side"] == "over" else -1)
            elif mk == "ml":
                c = last.get("ml_home") if b["side"] == "home" else last.get("ml_away")
                if c:
                    clv = (model.implied(c) - model.implied(b["odds"])) * 100  # prob points
            if clv is not None:
                o["clv_sum"] += clv
                o["clv_n"] += 1
                o["clv_pos"] += clv > 0
    for o in out.values():
        o["roi"] = o["units"] / o["n"] if o["n"] else None
        o["avg_clv"] = o["clv_sum"] / o["clv_n"] if o["clv_n"] else None
    return out


def update():
    now = store.now_et()
    today = now.date()
    cur = store.season_of(today)
    games = store.load_games()
    # history: two prior seasons (one to tune on, one to seed it) + this season to date
    hist_days = [d for s in (cur - 2, cur - 1) for d in store.season_days(s)]
    hist_days += [d for d in store.season_days(cur) if d < today - dt.timedelta(days=3)]
    print(f"Season {cur}: checking history...", flush=True)
    games = pull(hist_days, games)
    recent = [today + dt.timedelta(days=k) for k in range(-3, 2)]
    games = pull(recent, games, force={d.strftime("%Y%m%d") for d in recent})
    print(f"{len(games)} games stored, {len(store.finished(games))} final", flush=True)

    games, _ = backfill_lines(games, cur, budget=4500)

    # tune the first time, and again as soon as last season's lines are in (blend weights need them)
    meta = json.loads(store.PARAMS.read_text()) if store.PARAMS.exists() else {}
    last = store.finished(games)
    last_lines = int(last.loc[last["season"] == cur - 1, "spread"].notna().sum())
    if not meta or (not meta.get("blend_fit") and last_lines >= 3000):
        p = do_tune(games, cur)
        backtest.run_backtest(games, p, cur)
    p = load_params()

    now_utc = pd.Timestamp.now(tz="UTC")
    done = store.finished(games)
    pool = model.pooled_ids(done, p.pool_min)
    ratings = model.fit(done[pd.to_datetime(done["date"], utc=True) < now_utc], now_utc, cur, p, pool)
    picks, slate = update_picks(games, ratings, pool, p, now_utc)
    bt = json.loads(store.BACKTEST.read_text()) if store.BACKTEST.exists() else {}
    meta = json.loads(store.PARAMS.read_text()) if store.PARAMS.exists() else {}
    site.render(slate, ratings, record(picks, p), bt, p, now, calibrated=bool(meta.get("blend_fit")))
    print(f"Slate: {len(slate)} games priced. Site written.", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "update":
        update()
    elif cmd == "diag":
        from . import diag
        diag.main()
    elif cmd in ("tune", "backtest"):
        cur = store.season_of(store.now_et().date())
        games = store.load_games()
        p = do_tune(games, cur) if cmd == "tune" else load_params()
        rep = backtest.run_backtest(games, p, cur)
        print(json.dumps({k: v for k, v in rep.get("this_season", {}).items() if k != "buckets"}, indent=1))
        update()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

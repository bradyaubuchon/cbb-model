"""Entry point.  python -m cbb.run update|tune|backtest [ncaam,nba,nfl]"""
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

from . import backtest, espn, injuries, model, odds, site, store
from .sports import ORDER


def _fetched_file():
    return store.DATA / "fetched_days.txt"


def _fetched():
    f = _fetched_file()
    return set(f.read_text().split()) if f.exists() else set()


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
            _fetched_file().write_text("\n".join(sorted(done)))
            print(f"  ...{n} days fetched", flush=True)
        if not os.environ.get("CBB_FAKE"):
            time.sleep(0.4)  # ESPN rate-limits bursts
    games = store.upsert_games(games, rows)
    _fetched_file().write_text("\n".join(sorted(done)))
    return games


BOX_DONE = [False]
ABSENT = [{}]  # college: players who sat the team's last 2 games, by game id


def backfill_summaries(games, cur, budget):
    """Fill box scores (and closing lines where missing) for finished games from ESPN game
    summaries, a batch per run: last season first, then this season, then the season before."""
    SUMMARY_CHECKED = store.DATA / "summary_checked.txt"
    checked = set(SUMMARY_CHECKED.read_text().split()) if SUMMARY_CHECKED.exists() else set()
    for col in espn.BOX_COLS:
        if col not in games.columns:
            games[col] = float("nan")
    fin = store.finished(games)
    todo = fin[fin["h_fga"].isna() & (fin["season"] >= cur - 2) & ~fin["id"].isin(checked)]
    pri = todo["season"].map({cur - 1: 0, cur: 1, cur - 2: 2})
    todo = todo.assign(pri=pri).sort_values(["pri", "date"]).head(budget)
    if todo.empty:
        return games, 0
    remaining = int(len(fin[fin["h_fga"].isna() & (fin["season"] >= cur - 2) & ~fin["id"].isin(checked)])) - len(todo)
    print(f"Backfilling box scores/lines for {len(todo)} games "
          f"({int(fin['h_fga'].isna().sum())} without box scores)...", flush=True)
    idx = games.set_index("id")
    nbox = nline = 0
    for i, g in enumerate(todo.to_dict("records")):
        try:
            odds, box = espn.summary(g["id"], g["home_id"], g["away_id"], g["home_abbr"], g["away_abbr"])
        except Exception as e:
            print(f"  ! summary {g['id']}: {e}", flush=True)
            if "404" in str(e):
                checked.add(g["id"])  # ESPN has no page for this game; don't retry forever
            continue
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(g["date"])
        if box or age > pd.Timedelta(days=3):
            checked.add(g["id"])  # retry recent games whose box score isn't posted yet
        if box:
            for k, v in box.items():
                if k in idx.columns:
                    idx.at[g["id"], k] = v
            nbox += 1
        if odds and pd.isna(idx.at[g["id"], "spread"]) and pd.isna(idx.at[g["id"], "total"]):
            for k, v in odds.items():
                if v is not None:
                    idx.at[g["id"], k] = v
            nline += 1
        if not os.environ.get("CBB_FAKE"):
            time.sleep(0.2)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1} done ({nbox} box scores, {nline} new lines)", flush=True)
            idx.reset_index().to_csv(store.GAMES, index=False)
            SUMMARY_CHECKED.write_text("\n".join(sorted(checked)))
    games = idx.reset_index()
    games.to_csv(store.GAMES, index=False)
    SUMMARY_CHECKED.write_text("\n".join(sorted(checked)))
    print(f"  box scores {nbox}, new lines {nline}, {remaining} games left for later runs", flush=True)
    return games, remaining


def load_params():
    if store.PARAMS.exists():
        return model.Params.from_dict(json.loads(store.PARAMS.read_text()))
    return model.Params()


def tune_seasons(games, cur):
    """Last season, plus this season when last season has too few archived closing lines
    (ESPN's line archive starts in late Nov 2025, which leaves the 2025 NFL season thin)."""
    fin = store.finished(games)
    last_lines = int(fin.loc[fin["season"] == cur - 1, "spread"].notna().sum())
    return [cur - 1] if last_lines >= store.SPORT.min_lines_to_tune else [cur - 1, cur]


def do_tune(games, cur):
    seasons = tune_seasons(games, cur)
    print(f"Tuning both models on seasons {seasons} (walk-forward)...", flush=True)
    best, log = backtest.tune(games, seasons)
    pred = backtest.walk_forward(games, best, seasons, step=1)
    if not pred.empty and "inj_f" in pred.columns:
        f = pred["inj_f"].fillna(0).to_numpy()
        r = ((pred["home_score"] - pred["away_score"]) - pred["pred_margin"]).to_numpy()
        if (f != 0).sum() >= 200:
            best.inj_k = round(float((r * f).sum() / (f * f).sum()), 3)
            for c in ("pm_pts", "pm_eff", "pred_margin"):
                pred[c] = pred[c] + best.inj_k * f
            print(f"Injury effect: {best.inj_k} pts of margin per unit of missing value "
                  f"({int((f != 0).sum())} games with a rotation player out)", flush=True)
    mix = backtest.fit_mix(pred) if not pred.empty else {}
    if "margin" in mix:
        best.mix_margin, best.w_margin, best.margin_sd = mix["margin"]["mix"], mix["margin"]["w"], round(mix["margin"]["sd"], 2)
    if "total" in mix:
        best.mix_total, best.w_total, best.total_sd = mix["total"]["mix"], mix["total"]["w"], round(mix["total"]["sd"], 2)
    print("Mix/blend fit:", json.dumps(mix, indent=1), flush=True)
    if store.SPORT.key == "nfl" and not pred.empty:  # key numbers: 3, 7, 10...
        allp = backtest.walk_forward(games, best, [cur - 2, cur - 1, cur], step=3)
        if len(allp) >= 300:  # every game we have; market line as the expected margin where known
            mean = (-allp["spread"]).fillna(allp["pred_margin"])
            best.key_w = model.fit_key_weights((allp["home_score"] - allp["away_score"]).to_numpy(),
                                               mean.to_numpy(), best.margin_sd, smooth=15.0)
            print("NFL key-number weights:", {k: v for k, v in best.key_w.items() if v > 1.2}, flush=True)
    has_box = "h_fga" in games.columns and int(store.finished(games).query("season == @cur - 1")["h_fga"].notna().sum())
    store.PARAMS.write_text(json.dumps({**best.to_dict(), "blend_fit": "margin" in mix, "box_games_at_tune": has_box, "box_complete": BOX_DONE[0], "inj_fit": "inj_f" in games.columns,
                                        "mix_fit": mix, "tune_log": log}, indent=1, default=float))
    print("Chosen:", best.to_dict(), flush=True)
    return best


def update_picks(games, ratings, pool, p, now_utc):
    picks = json.loads(store.PICKS.read_text()) if store.PICKS.exists() else {}
    t = pd.to_datetime(games["date"], utc=True)
    upcoming = games[(games["state"] == "pre") & (t > now_utc) & (t < now_utc + pd.Timedelta(hours=store.SPORT.horizon_h))]
    slate = []
    inj_f, inj_detail = {}, {}
    if store.SPORT.key == "nba" and not upcoming.empty:
        inj_f, inj_detail = injuries.upcoming_missing(games, upcoming.to_dict("records"))
    elif store.SPORT.key == "ncaam" and "inj_f" in games.columns:
        inj_f = games.set_index("id")["inj_f"].fillna(0).to_dict()
        inj_detail = ABSENT[0]
    for g in upcoming.to_dict("records"):
        mg, tot, hp, ap = ratings.predict(g["home_id"], g["away_id"], g["neutral"] == 1, pool, g.get("rest_f"))
        adj = p.inj_k * inj_f.get(g["id"], 0.0)
        mg, hp, ap = mg + adj, hp + adj / 2, ap - adj / 2
        priced = model.price_game(mg, tot, g, p)
        snap = {"time": now_utc.isoformat(), "pred_margin": mg, "pred_total": tot, "bets": priced,
                "spread": g.get("spread"), "total": g.get("total"),
                "ml_home": g.get("ml_home"), "ml_away": g.get("ml_away")}
        rec = picks.get(g["id"])
        if priced and (rec is None or "first" not in rec):
            rec = picks.setdefault(g["id"], {"id": g["id"], "date": g["date"], "home": g["home"], "away": g["away"]})
            rec["first"] = snap
        if rec is not None:
            rec["last"] = snap
        slate.append({**{k: g.get(k) for k in ("id", "date", "home", "away", "home_abbr", "away_abbr", "home_full", "away_full", "neutral", "spread", "total",
                                               "ml_home", "ml_away", "book", "spread_open", "total_open")},
                      "pred_home": hp, "pred_away": ap, "pred_margin": mg, "pred_total": tot,
                      "blend_margin": model.blend(mg, tot, g, p)[0], "blend_total": model.blend(mg, tot, g, p)[1],
                      "home_games": ratings.n_games.get(g["home_id"], 0),
                      "away_games": ratings.n_games.get(g["away_id"], 0),
                      "rest_h": g.get("rest_h"), "rest_a": g.get("rest_a"),
                      "injuries": inj_detail.get(g["id"], []), "inj_adj": adj,
                      "bets": priced})
    # attach final scores
    fin = store.finished(games).set_index("id")
    for gid, rec in picks.items():
        if gid in fin.index and "final" not in rec:
            rec["final"] = [float(fin.at[gid, "home_score"]), float(fin.at[gid, "away_score"])]
    store.PICKS.write_text(json.dumps(picks, indent=0, default=float))
    return picks, sorted(slate, key=lambda s: s["date"])


def log_market(slate, picks):
    """Log each price edge the first time it's flagged (what you could have bet), and keep the
    latest fair line before tip so closing value can be measured."""
    for g in slate:
        mk = g.get("market")
        if not mk:
            continue
        rec = picks.setdefault(g["id"], {"id": g["id"], "date": g["date"], "home": g["home"], "away": g["away"]})
        first = rec.setdefault("mkt_first", {})
        for m, b in mk["best"].items():
            if b["ev"] >= odds.MIN_EV and m not in first:
                first[m] = {**b, "time": mk["snap_time"]}
        rec["mkt_fair_last"] = mk["fair"]
    store.PICKS.write_text(json.dumps(picks, indent=0, default=float))


def _fair_p(b, fv, p):
    """Chance the logged bet wins under a fair line (for closing value)."""
    if not fv:
        return None
    if b["market"] == "spread" and fv.get("margin") is not None:
        hp = b["line"] if b["side"] == "home" else -b["line"]
        w, push = model.margin_probs(fv["margin"], p.margin_sd, -hp, p.key_w)
        return w if b["side"] == "home" else 1 - w - push
    if b["market"] == "total" and fv.get("total") is not None:
        w, push = model.prob_over(fv["total"], p.total_sd, b["line"])
        return w if b["side"] == "over" else 1 - w - push
    if b["market"] == "ml" and fv.get("home_p") is not None:
        return fv["home_p"] if b["side"] == "home" else 1 - fv["home_p"]
    return None


def record_market(picks, p):
    out = {mk: {"n": 0, "w": 0, "l": 0, "push": 0, "units": 0.0, "risked": 0.0, "clv_sum": 0.0, "clv_n": 0, "clv_pos": 0}
           for mk in ("spread", "total", "ml")}
    for rec in picks.values():
        if "final" not in rec or not rec.get("mkt_first"):
            continue
        hs, as_ = rec["final"]
        for mk, b in rec["mkt_first"].items():
            o = out[mk]
            prof = model.grade(b, mk, hs, as_)
            o["n"] += 1
            o["units"] += prof
            o["risked"] += model.stake(b["odds"])
            o["w" if prof > 0 else "l" if prof < 0 else "push"] += 1
            pl = _fair_p(b, rec.get("mkt_fair_last"), p)
            if pl is not None:
                clv = (pl - b["p"]) * 100  # change in fair win chance, percentage points
                o["clv_sum"] += clv
                o["clv_n"] += 1
                o["clv_pos"] += clv > 0
    for o in out.values():
        o["roi"] = o["units"] / o["risked"] if o["risked"] else None
        o["avg_clv"] = o["clv_sum"] / o["clv_n"] if o["clv_n"] else None
    return out


def record(picks, p):
    """Live results of flagged plays. 'first' = the price when the model first saw the game
    (what you could have bet); CLV compares it to the last line before tip."""
    out = {mk: {"n": 0, "w": 0, "l": 0, "push": 0, "units": 0.0, "risked": 0.0, "clv_sum": 0.0, "clv_n": 0, "clv_pos": 0}
           for mk in ("spread", "total", "ml")}
    for rec in picks.values():
        if "final" not in rec or "first" not in rec:  # price-edge-only games have no model snapshot
            continue
        hs, as_ = rec["final"]
        for mk in out:
            b = rec["first"]["bets"].get(mk)
            if not b or b["ev"] < p.min_ev or (mk == "ml" and not p.ml_flag):
                continue
            o = out[mk]
            prof = model.grade(b, mk, hs, as_)
            o["n"] += 1
            o["units"] += prof
            o["risked"] += model.stake(b["odds"])
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
        o["roi"] = o["units"] / o["risked"] if o["risked"] else None
        o["avg_clv"] = o["clv_sum"] / o["clv_n"] if o["clv_n"] else None
    return out


def update(sport, budget):
    store.set_sport(sport)
    print(f"\n===== {store.SPORT.name} =====", flush=True)
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

    if games.empty:
        print("No games yet.", flush=True)
        return None
    games, box_left = backfill_summaries(games, cur, budget=budget)
    games = model.add_rest(games, store.SPORT)
    players_left = None
    if sport == "nba":
        players_left = injuries.backfill(games, cur, budget=1500)
        miss = injuries.missing_by_game(games)
        games["inj_f"] = games["id"].map(miss)
    elif sport == "ncaam":
        players_left = injuries.backfill(games, cur, budget=1500)
        miss, ABSENT[0] = injuries.prior_absence(games)
        games["inj_f"] = games["id"].map(miss)

    # tune the first time, and again as soon as last season's lines are in (blend weights need them)
    meta = json.loads(store.PARAMS.read_text()) if store.PARAMS.exists() else {}
    last = store.finished(games)
    last_lines = int(last.loc[last["season"].isin(tune_seasons(games, cur)), "spread"].notna().sum())
    retune = not meta or (not meta.get("blend_fit") and last_lines >= store.SPORT.min_lines_to_tune) \
        or (box_left == 0 and not meta.get("box_complete")) \
        or (players_left == 0 and not meta.get("inj_fit"))  # player data complete: fit the absence effect
    BOX_DONE[0] = box_left == 0
    if retune:
        p = do_tune(games, cur)
        backtest.run_backtest(games, p, cur)
    p = load_params()

    now_utc = pd.Timestamp.now(tz="UTC")
    done = store.finished(games)
    pool = model.pooled_ids(done, p.pool_min)
    ratings = model.fit_all(done[pd.to_datetime(done["date"], utc=True) < now_utc], now_utc, cur, p, pool)
    picks, slate = update_picks(games, ratings, pool, p, now_utc)
    snap_time = None
    if slate:
        try:
            snap_time = odds.apply(slate, p, 3, now)
            log_market(slate, picks)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ! sportsbook comparison failed: {e}", flush=True)
    bt = json.loads(store.BACKTEST.read_text()) if store.BACKTEST.exists() else {}
    meta = json.loads(store.PARAMS.read_text()) if store.PARAMS.exists() else {}
    rec = {**record(picks, p), "price": record_market(picks, p)}
    ver = site.render(sport, slate, ratings, rec, bt, p, now, calibrated=bool(meta.get("blend_fit")),
                      odds_time=snap_time)
    print(f"Slate: {len(slate)} games priced. Site written.", flush=True)
    return ver


def run_all(cmd, only=None):
    """Each sport runs on its own; one sport failing never blocks the others."""
    store.migrate_legacy()
    versions = {}
    # ESPN summary fetches are the slow part (~0.5s each): share one budget per run
    budget = {"ncaam": 4000, "nba": 2500, "nfl": 1500}
    for sport in (only or ORDER):
        try:
            if cmd in ("tune", "backtest"):
                store.set_sport(sport)
                games = model.add_rest(store.load_games(), store.SPORT)
                if sport == "nba":
                    games["inj_f"] = games["id"].map(injuries.missing_by_game(games))
                elif sport == "ncaam":
                    games["inj_f"] = games["id"].map(injuries.prior_absence(games)[0])
                if not games.empty:
                    cur = store.season_of(store.now_et().date())
                    p = do_tune(games, cur) if cmd == "tune" else load_params()
                    backtest.run_backtest(games, p, cur)
            versions[sport] = update(sport, budget[sport])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {sport} failed: {e}", flush=True)
    site.write_version(versions)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if cmd == "diag":
        from . import diag
        diag.main()
    elif cmd in ("update", "tune", "backtest"):
        run_all(cmd, only)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

"""NBA player availability.

1. Player box scores come from ESPN game summaries (data/nba/players.csv).
2. Each player's value is his recent production per minute (Hollinger Game Score per 36,
   above a replacement level) times the minutes he usually plays - using only games before
   the one being predicted.
3. A team's 'missing value' for a game = value of its regular rotation players who didn't play.
   In past games that's read off the box score; for upcoming games, from ESPN's injury report.
4. The backtest fits how many points of margin one unit of missing value is worth (inj_k).
"""
import json

import numpy as np
import pandas as pd

from . import espn, store

PLAYER_COLS = ["game_id", "team_id", "player_id", "name", "dnp", "min", "pts", "fgm", "fga", "ftm", "fta",
               "oreb", "dreb", "ast", "stl", "blk", "to", "pf"]
REPL = 8.0       # Game Score per 36 minutes of a replacement-level player
ROTATION_MIN = 14.0
WINDOW = 15      # games used for a player's recent form


def players_file():
    return store.DATA / "players.csv"


def checked_file():
    return store.DATA / "players_checked.txt"


def _n(x):
    try:
        return float(str(x).split("-")[0]) if x not in (None, "", "--") else 0.0
    except ValueError:
        return 0.0


def parse_players(d, gid):
    rows = []
    for t in (d.get("boxscore") or {}).get("players") or []:
        tid = str((t.get("team") or {}).get("id"))
        st = (t.get("statistics") or [{}])[0]
        keys = st.get("keys") or []
        for a in st.get("athletes") or []:
            v = dict(zip(keys, a.get("stats") or []))
            fg = str(v.get("fieldGoalsMade-fieldGoalsAttempted", "0-0")).split("-")
            ft = str(v.get("freeThrowsMade-freeThrowsAttempted", "0-0")).split("-")
            ath = a.get("athlete") or {}
            rows.append({
                "game_id": gid, "team_id": tid, "player_id": str(ath.get("id")), "name": ath.get("displayName"),
                "dnp": int(bool(a.get("didNotPlay")) or not a.get("stats")),
                "min": _n(v.get("minutes")), "pts": _n(v.get("points")),
                "fgm": _n(fg[0]), "fga": _n(fg[1] if len(fg) > 1 else 0), "ftm": _n(ft[0]),
                "fta": _n(ft[1] if len(ft) > 1 else 0), "oreb": _n(v.get("offensiveRebounds")),
                "dreb": _n(v.get("defensiveRebounds")), "ast": _n(v.get("assists")), "stl": _n(v.get("steals")),
                "blk": _n(v.get("blocks")), "to": _n(v.get("turnovers")), "pf": _n(v.get("fouls")),
            })
    return rows


def backfill(games, cur, budget):
    """Player box scores for finished games (this season and last two)."""
    import os
    import time
    ck = set(checked_file().read_text().split()) if checked_file().exists() else set()
    fin = store.finished(games)
    todo = fin[(fin["season"] >= cur - 2) & ~fin["id"].isin(ck)].sort_values("date", ascending=False).head(budget)
    if todo.empty:
        return
    print(f"Player box scores: fetching {len(todo)} games", flush=True)
    rows = []
    for i, g in enumerate(todo.to_dict("records")):
        try:
            if os.environ.get("CBB_FAKE"):
                from tests.fake_espn import fake_players
                d = fake_players(g)
            else:
                d = espn._get_json(f"{espn.API}/{store.SPORT.espn}/summary?event={g['id']}")
            r = parse_players(d, g["id"])
        except Exception as e:
            print(f"  ! players {g['id']}: {e}", flush=True)
            if "404" in str(e):
                ck.add(g["id"])
            continue
        if r:
            rows += r
            ck.add(g["id"])
        elif (pd.Timestamp.now(tz="UTC") - pd.Timestamp(g["date"])).days > 3:
            ck.add(g["id"])
        if not os.environ.get("CBB_FAKE"):
            time.sleep(0.2)
        if (i + 1) % 500 == 0:
            _save(rows, ck)
            rows = []
            print(f"  ...{i + 1}", flush=True)
    _save(rows, ck)


def _save(rows, ck):
    if rows:
        new = pd.DataFrame(rows, columns=PLAYER_COLS)
        f = players_file()
        if f.exists():
            old = pd.read_csv(f, dtype={"game_id": str, "team_id": str, "player_id": str})
            new = pd.concat([old[~old["game_id"].isin(new["game_id"])], new])
        new.to_csv(f, index=False)
    checked_file().write_text("\n".join(sorted(ck)))


def load_players():
    f = players_file()
    if not f.exists():
        return pd.DataFrame(columns=PLAYER_COLS)
    return pd.read_csv(f, dtype={"game_id": str, "team_id": str, "player_id": str})


def _game_score(p):
    return (p["pts"] + 0.4 * p["fgm"] - 0.7 * p["fga"] - 0.4 * (p["fta"] - p["ftm"]) + 0.7 * p["oreb"]
            + 0.3 * p["dreb"] + p["stl"] + 0.7 * p["ast"] + 0.7 * p["blk"] - 0.4 * p["pf"] - p["to"])


def history(games):
    """One row per player per team game (including games he missed), in date order, with his
    value going INTO that game computed from earlier games only."""
    pl = load_players()
    if pl.empty:
        return pl
    dates = games.set_index("id")["date"]
    pl = pl[pl["game_id"].isin(dates.index)].copy()
    pl["t"] = pd.to_datetime(pl["game_id"].map(dates), utc=True)
    pl["played"] = (pl["dnp"] == 0) & (pl["min"] > 0)
    pl["gs"] = np.where(pl["played"], _game_score(pl), np.nan)
    pl["mn"] = np.where(pl["played"], pl["min"], 0.0)
    pl = pl.sort_values(["player_id", "t"])
    g = pl.groupby("player_id")
    # rolling over the previous WINDOW appearances on any team, shifted so a game never sees itself
    pl["min_avg"] = g["mn"].transform(lambda s: s.shift().rolling(WINDOW, min_periods=3).mean())
    gs_sum = g["gs"].transform(lambda s: s.fillna(0).shift().rolling(WINDOW, min_periods=3).sum())
    mn_sum = g["mn"].transform(lambda s: s.shift().rolling(WINDOW, min_periods=3).sum())
    pl["gs36"] = 36 * gs_sum / mn_sum.replace(0, np.nan)
    pl["value"] = (pl["gs36"] - REPL).clip(lower=0) * pl["min_avg"] / 48
    return pl


def missing_by_game(games):
    """{game_id: inj_f} where inj_f = away missing value - home missing value (positive helps
    the home team). Only rotation players (usual minutes >= ROTATION_MIN) count."""
    h = history(games)
    if h.empty:
        return {}
    rot = h[(h["min_avg"] >= ROTATION_MIN) & ~h["played"]]
    miss = rot.groupby(["game_id", "team_id"])["value"].sum()
    out = {}
    g = games[games["id"].isin(h["game_id"].unique())]
    for gid, hid, aid in zip(g["id"], g["home_id"], g["away_id"]):
        out[gid] = float(miss.get((gid, str(aid)), 0.0) - miss.get((gid, str(hid)), 0.0))
    return out


def fetch_report():
    """{team display name: [(player name, status)]} from ESPN's injury report."""
    import os
    if os.environ.get("CBB_FAKE"):
        from tests.fake_espn import fake_injuries
        d = fake_injuries()
    else:
        d = espn._get_json(f"{espn.API}/{store.SPORT.espn}/injuries")
    out = {}
    for t in d.get("injuries") or []:
        out[t.get("displayName")] = [((i.get("athlete") or {}).get("displayName"), i.get("status"))
                                     for i in t.get("injuries") or []]
    return out


STATUS_W = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.4, "Day-To-Day": 0.4}


def upcoming_missing(games, upcoming):
    """For upcoming games: (inj_f per game id, {game id: [(team, player, status, value)]})."""
    h = history(games)
    if h.empty:
        return {}, {}
    try:
        report = fetch_report()
    except Exception as e:
        print(f"  ! injury report failed: {e}", flush=True)
        return {}, {}
    pl_all = h.sort_values("t")
    # current value: recompute from each player's most recent WINDOW appearances
    cur = {}
    for pid, grp in pl_all[pl_all["played"]].groupby("player_id"):
        grp = grp.tail(WINDOW)
        mins = grp["mn"].mean()
        gs36 = 36 * grp["gs"].sum() / max(1.0, grp["mn"].sum())
        cur[pid] = (max(0.0, gs36 - REPL) * mins / 48, mins, grp["team_id"].iloc[-1], grp["name"].iloc[-1])
    by_team_name = {}
    names = dict(zip(games["home_id"].astype(str), games.get("home_full", games["home"])))
    names.update(dict(zip(games["away_id"].astype(str), games.get("away_full", games["away"]))))
    for pid, (val, mins, tid, nm) in cur.items():
        if mins >= ROTATION_MIN:
            by_team_name.setdefault((names.get(tid), nm), (val, tid))
    f, detail = {}, {}
    for g in upcoming:
        tot = {"home": 0.0, "away": 0.0}
        rows = []
        for side in ("home", "away"):
            tname = g.get(f"{side}_full") or g[side]
            for player, status in report.get(tname, []):
                w = STATUS_W.get(status, 0.0)
                hit = by_team_name.get((tname, player))
                if hit and w > 0:
                    tot[side] += w * hit[0]
                    rows.append({"team": g[side], "player": player, "status": status, "value": round(hit[0], 2)})
        f[g["id"]] = tot["away"] - tot["home"]
        if rows:
            detail[g["id"]] = sorted(rows, key=lambda r: -r["value"])
    return f, detail

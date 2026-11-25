"""Sportsbook comparison via The Odds API (free tier: 500 credits/month).

Idea: Pinnacle is the sharpest book, so its prices (with the vig removed) are the best
available estimate of each outcome's true chance. When a US book you can bet at pays more
than that fair price, the difference is the edge. No ratings model is involved.

Spreads and totals at different numbers are compared by converting Pinnacle's price into an
implied mean margin / total, then pricing the other book's number from that mean.
"""
import datetime as dt
import json
import os
import re
import urllib.request

import pandas as pd

from . import model, store

API = "https://api.the-odds-api.com/v4/sports"
MARKETS, REGIONS = "h2h,spreads,totals", "us,eu"
COST = 6               # credits per call = markets x regions
MIN_EV = 0.02          # flag a price edge at +2% EV
SHARP = "pinnacle"
# books a US bettor can actually use (regulated + common offshore in the 'us' region)
BETTABLE = {"draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "betrivers", "fanatics",
            "hardrockbet", "ballybet", "betparx", "fliff", "bovada", "betonlineag", "lowvig",
            "mybookieag", "betus", "superbook", "wynnbet", "unibet_us", "pointsbetus", "twinspires"}


def state_file():
    return store.ROOT / "data" / "odds_state.json"


def load_state():
    f = state_file()
    return json.loads(f.read_text()) if f.exists() else {"last": {}}


def save_state(s):
    state_file().write_text(json.dumps(s, indent=1))


def should_fetch(sport, n_active, now):
    """Spread the monthly credits evenly over the days left in the month across active sports."""
    if not os.environ.get("ODDS_API_KEY"):
        return False, "no ODDS_API_KEY secret"
    st = load_state()
    rem = st.get("remaining")
    last = st["last"].get(sport)
    if rem is None or last is None:
        return True, "first snapshot"
    if rem < COST * 2:
        return False, f"only {rem} credits left this month"
    nxt = (now.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    days_left = max(1.0, (nxt - now).total_seconds() / 86400)
    per_day = rem / days_left / max(1, n_active) / COST       # snapshots/day this sport can afford
    interval_h = min(24.0, max(2.0, 24.0 / max(per_day, 0.01)))
    age_h = (now - dt.datetime.fromisoformat(last)).total_seconds() / 3600
    if age_h >= interval_h - 0.25:
        return True, f"every {interval_h:.1f}h on budget"
    return False, f"next snapshot in {interval_h - age_h:.1f}h ({rem} credits left)"


def fetch(sport_key):
    if os.environ.get("CBB_FAKE"):
        from tests.fake_espn import fake_odds
        return fake_odds(store.SPORT), 480
    url = (f"{API}/{sport_key}/odds?apiKey={os.environ['ODDS_API_KEY']}&regions={REGIONS}"
           f"&markets={MARKETS}&oddsFormat=american&dateFormat=iso")
    req = urllib.request.Request(url, headers={"User-Agent": "personal-betting-model"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
        rem = r.headers.get("x-requests-remaining")
    return data, (int(float(rem)) if rem is not None else None)


# ---------- matching ESPN games to Odds API events ----------

def _tokens(name):
    s = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower().replace("&", " and "))
    s = s.replace(" st ", " state ").replace("saint ", "st ")
    return set(s.split())


def _sim(a, b):
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def match(slate, events):
    """{espn_id: odds_event}. Start times within 4h, both team names must agree closely
    (so 'Los Angeles Rams' never pairs with 'Los Angeles Chargers'), and each event is used once."""
    cands = []
    for g in slate:
        t = pd.Timestamp(g["date"])
        for e in events:
            if abs((pd.Timestamp(e["commence_time"]) - t).total_seconds()) > 4 * 3600:
                continue
            sc = min(_sim(g.get("home_full") or g["home"], e["home_team"]),
                     _sim(g.get("away_full") or g["away"], e["away_team"]))
            if sc >= 0.6:
                cands.append((sc, g["id"], e["id"], e))
    out, used = {}, set()
    for sc, gid, eid, e in sorted(cands, key=lambda c: -c[0]):
        if gid not in out and eid not in used:
            out[gid] = e
            used.add(eid)
    return out


# ---------- fair prices ----------

def _devig(a, b):
    ia, ib = model.implied(a), model.implied(b)
    return ia / (ia + ib)


def _solve(prob_fn, target, lo=-80.0, hi=200.0):
    """Mean m such that prob_fn(m) == target (prob_fn increasing in m)."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if prob_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _two_way(book, mkey, name_a):
    """(outcome named name_a, the other outcome) for a two-way market, or None."""
    for m in book.get("markets", []):
        if m["key"] == mkey and len(m.get("outcomes", [])) == 2:
            o = m["outcomes"]
            a = next((x for x in o if x["name"] == name_a), None)
            b = next((x for x in o if x["name"] != name_a), None)
            if a and b:
                return a, b
    return None


def _cover(mean, home_point, p):
    """P(home covers), P(push) at home_point given mean margin."""
    return model.margin_probs(mean, p.margin_sd, -home_point, p.key_w)


def fair(event, p):
    """Fair mean margin (home - away), fair mean total, fair home win chance, and the source."""
    home = event["home_team"]
    books = {b["key"]: b for b in event.get("bookmakers", [])}
    srcs = [books[SHARP]] if SHARP in books else list(books.values())
    mus, taus, qs = [], [], []
    for b in srcs:
        sp = _two_way(b, "spreads", home)
        if sp:
            q = _devig(sp[0]["price"], sp[1]["price"])
            pt = sp[0]["point"]
            mus.append(_solve(lambda m: (lambda w, ps: w / max(1e-9, 1 - ps))(*_cover(m, pt, p)), q, -80, 80))
        to = _two_way(b, "totals", "Over")
        if to:
            q = _devig(to[0]["price"], to[1]["price"])
            pt = to[0]["point"]
            taus.append(_solve(lambda m: (lambda w, ps: w / max(1e-9, 1 - ps))(*model.prob_over(m, p.total_sd, pt)), q, 0, 400))
        ml = _two_way(b, "h2h", home)
        if ml:
            qs.append(_devig(ml[0]["price"], ml[1]["price"]))
    med = lambda v: float(pd.Series(v).median()) if v else None
    return {"margin": med(mus), "total": med(taus), "home_p": med(qs),
            "source": "Pinnacle" if SHARP in books else f"consensus of {len(books)} books"}


def edges(event, fv, p):
    """Every bettable price, valued against the fair line. Returns them best first."""
    home, away = event["home_team"], event["away_team"]
    out = []
    for b in event.get("bookmakers", []):
        if b["key"] not in BETTABLE:
            continue
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                price, pt = o.get("price"), o.get("point")
                if price is None:
                    continue
                if m["key"] == "spreads" and fv["margin"] is not None and pt is not None:
                    side = "home" if o["name"] == home else "away"
                    hp = pt if side == "home" else -pt
                    w, push = _cover(fv["margin"], hp, p)
                    pw = w if side == "home" else 1 - w - push
                    mk, line = "spread", pt
                elif m["key"] == "totals" and fv["total"] is not None and pt is not None:
                    side = "over" if o["name"] == "Over" else "under"
                    w, push = model.prob_over(fv["total"], p.total_sd, pt)
                    pw = w if side == "over" else 1 - w - push
                    mk, line = "total", pt
                elif m["key"] == "h2h" and fv["home_p"] is not None and o["name"] in (home, away):
                    side = "home" if o["name"] == home else "away"
                    pw, push = (fv["home_p"] if side == "home" else 1 - fv["home_p"]), 0.0
                    mk, line = "ml", None
                else:
                    continue
                out.append({"market": mk, "side": side, "line": line, "odds": float(price), "p": pw, "push": push,
                            "ev": model.ev(pw, push, float(price)), "book": b["key"], "book_title": b["title"]})
    return sorted(out, key=lambda x: -x["ev"])


def best_by_market(all_edges):
    """Top price for each market (one bet per market per game)."""
    best = {}
    for e in all_edges:
        if e["ev"] > 0.15:  # real mispricings are small; this is almost always a stale or mismatched line
            continue
        if e["market"] not in best:
            best[e["market"]] = e
    return best


def apply(slate, p, n_active, now):
    """Fetch (if the budget allows), value every price, attach results to the slate and log
    flagged price edges for grading. Snapshot is kept so runs between fetches still show it."""
    sport = store.SPORT.key
    snap_f = store.DATA / "odds_latest.json"
    go, why = should_fetch(sport, n_active, now)
    print(f"Odds API: {'fetching' if go else 'skip'} ({why})", flush=True)
    if go:
        try:
            events, rem = fetch(store.SPORT.odds_key)
            st = load_state()
            st["last"][sport] = now.isoformat()
            if rem is not None:
                st["remaining"] = rem
            save_state(st)
            snap_f.write_text(json.dumps({"time": now.isoformat(), "events": events}))
            print(f"  {len(events)} events, {rem} credits left", flush=True)
        except Exception as e:
            print(f"  ! odds fetch failed: {e}", flush=True)
    if not snap_f.exists():
        return None
    snap = json.loads(snap_f.read_text())
    m = match(slate, snap["events"])
    for g in slate:
        e = m.get(g["id"])
        if not e:
            continue
        fv = fair(e, p)
        ed = edges(e, fv, p)
        g["market"] = {"fair": fv, "best": best_by_market(ed), "snap_time": snap["time"],
                       "books": len(e.get("bookmakers", []))}
    print(f"  matched {len(m)}/{len(slate)} games to sportsbook odds", flush=True)
    return snap["time"]

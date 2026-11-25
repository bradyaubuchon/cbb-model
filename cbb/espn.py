"""ESPN public scoreboard: fetch one day of D-I men's college basketball and flatten it.

ESPN's spread is the HOME team's line (negative = home favored). Odds come from whichever
book ESPN shows (DraftKings / ESPN BET). Fields have moved around over the years, so every
value is read from several possible places and cross-checked.
"""
import json
import os
import time
import urllib.request

from . import store

API = "https://site.api.espn.com/apis/site/v2/sports"
LIMIT = 500  # ESPN silently truncates above ~1000; a day that fills the limit is an error
HEADERS = {"User-Agent": "Mozilla/5.0 (personal cbb model)", "Accept": "application/json"}


def _get_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def scoreboard(day):
    """Raw events for one ET calendar day, day = 'YYYYMMDD'."""
    if os.environ.get("CBB_FAKE"):
        from tests.fake_espn import fake_day
        return fake_day(day, store.SPORT)
    sp = store.SPORT
    data = _get_json(f"{API}/{sp.espn}/scoreboard?dates={day}{sp.query}&limit={LIMIT}")
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise RuntimeError(f"ESPN scoreboard {day}: no events list")
    if len(events) >= LIMIT:
        raise RuntimeError(f"ESPN scoreboard {day}: hit limit {LIMIT}, games may be missing")
    return events


# ---------- parsing helpers ----------

def _num(x):
    """'-7.5' -> -7.5, 'o145.5' -> 145.5, 'EVEN'/'PK' -> 0, '+140' -> 140, junk -> None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK", "PICK"):
        return 0.0
    s = s.lstrip("OU").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _pick_odds(comp):
    odds = comp.get("odds") or []
    if not odds:
        return {}
    for pref in ("draftkings", "espn bet", "espnbet"):
        for o in odds:
            name = str(_dig(o, "provider", "name") or "").lower()
            if pref in name:
                return o
    return odds[0]


def parse_odds(o, home_abbr, away_abbr):
    """One provider's odds block (scoreboard 'odds' and summary 'pickcenter' share a shape)."""
    o = o or {}
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    spread = _num(_first(o.get("spread"), _dig(o, "pointSpread", "home", "close", "line")))
    details = str(o.get("details") or "")
    # cross-check sign against 'ABBR -7.5' text; trust the text if they disagree
    parts = details.split()
    if len(parts) == 2 and _num(parts[1]) is not None:
        fav_line = -abs(_num(parts[1]))
        if parts[0] == home_abbr:
            spread = fav_line
        elif parts[0] == away_abbr:
            spread = -fav_line
    elif details.upper() in ("EVEN", "PK", "PICK"):
        spread = 0.0
    row = {
        "book": _dig(o, "provider", "name"),
        "spread": spread,
        "spread_open": _num(_dig(o, "pointSpread", "home", "open", "line")),
        "spread_home_odds": _num(_first(ho.get("spreadOdds"), _dig(o, "pointSpread", "home", "close", "odds"))),
        "spread_away_odds": _num(_first(ao.get("spreadOdds"), _dig(o, "pointSpread", "away", "close", "odds"))),
        "total": _num(_first(o.get("overUnder"), _dig(o, "total", "over", "close", "line"))),
        "total_open": _num(_dig(o, "total", "over", "open", "line")),
        "over_odds": _num(_first(o.get("overOdds"), _dig(o, "total", "over", "close", "odds"))),
        "under_odds": _num(_first(o.get("underOdds"), _dig(o, "total", "under", "close", "odds"))),
        "ml_home": _num(_first(ho.get("moneyLine"), _dig(o, "moneyline", "home", "close", "odds"))),
        "ml_away": _num(_first(ao.get("moneyLine"), _dig(o, "moneyline", "away", "close", "odds"))),
    }
    # anything between -100 and +100 is not a real American price
    for k in ("ml_home", "ml_away", "spread_home_odds", "spread_away_odds", "over_odds", "under_odds"):
        if row[k] is not None and abs(row[k]) < 100:
            row[k] = None
    return row




BOX_STATS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb",),
    "totalTurnovers": ("to",),
}
BOX_COLS = [f"{side}_{k}" for side in ("h", "a") for k in ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "to")]


def _box_team(t):
    out = {}
    for st in t.get("statistics") or []:
        keys = BOX_STATS.get(st.get("name"))
        if not keys:
            continue
        parts = str(st.get("displayValue", "")).split("-")
        if len(parts) != len(keys):
            continue
        for k, v in zip(keys, parts):
            out[k] = _num(v)
    if "to" not in out:  # older feeds: plain 'turnovers'
        for st in t.get("statistics") or []:
            if st.get("name") == "turnovers":
                out["to"] = _num(st.get("displayValue"))
    return out


def summary(event_id, home_id, away_id, home_abbr, away_abbr):
    """(closing odds, box score) for a finished game. The scoreboard drops odds after tip;
    the game summary keeps them under 'pickcenter' along with team box scores."""
    if os.environ.get("CBB_FAKE"):
        from tests.fake_espn import fake_summary
        d = fake_summary(event_id, store.SPORT)
    else:
        d = _get_json(f"{API}/{store.SPORT.espn}/summary?event={event_id}")
    d = d or {}
    pc = d.get("pickcenter") or []
    odds = parse_odds(_pick_odds({"odds": pc}), home_abbr, away_abbr) if pc else None
    box = {}
    for t in (d.get("boxscore") or {}).get("teams") or []:
        tid = str(_dig(t, "team", "id"))
        side = "h" if tid == str(home_id) else "a" if tid == str(away_id) else None
        if side:
            for k, v in _box_team(t).items():
                box[f"{side}_{k}"] = v
    need = ("h_fga", "h_fta", "h_oreb", "h_to", "a_fga", "a_fta", "a_oreb", "a_to")
    if not all(box.get(k) is not None for k in need):
        box = None
    return odds, box


def parse_event(e):
    stype = _dig(e, "season", "type")
    if stype is not None and stype not in (2, 3):  # regular season and postseason only
        return None
    comp = (e.get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    h, a = teams.get("home"), teams.get("away")
    if not h or not a:
        return None
    status = comp.get("status") or e.get("status") or {}
    state = _dig(status, "type", "state") or "pre"
    row = {
        "id": str(e.get("id")),
        "date": e.get("date"),
        "season": _dig(e, "season", "year"),
        "state": state,
        "neutral": int(bool(comp.get("neutralSite"))),
        "conf_game": int(bool(comp.get("conferenceCompetition"))),
        "home_id": str(_dig(h, "team", "id")),
        "home": _dig(h, "team", "shortDisplayName") or _dig(h, "team", "displayName"),
        "home_abbr": _dig(h, "team", "abbreviation"),
        "away_id": str(_dig(a, "team", "id")),
        "away": _dig(a, "team", "shortDisplayName") or _dig(a, "team", "displayName"),
        "away_abbr": _dig(a, "team", "abbreviation"),
        "home_score": None,
        "away_score": None,
    }
    if state == "post" and _dig(status, "type", "completed"):
        row["home_score"] = _num(h.get("score"))
        row["away_score"] = _num(a.get("score"))
    elif state == "post":
        row["state"] = "canceled"  # postponed / canceled / forfeit

    row.update(parse_odds(_pick_odds(comp), row["home_abbr"], row["away_abbr"]))
    return row


def parse_day(events):
    out, seen = [], set()
    for e in events:
        try:
            r = parse_event(e)
        except Exception:
            r = None
        if r and r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out

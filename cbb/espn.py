"""ESPN public scoreboard: fetch one day of D-I men's college basketball and flatten it.

ESPN's spread is the HOME team's line (negative = home favored). Odds come from whichever
book ESPN shows (DraftKings / ESPN BET). Fields have moved around over the years, so every
value is read from several possible places and cross-checked.
"""
import json
import os
import time
import urllib.request

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
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
        return fake_day(day)
    data = _get_json(f"{BASE}?dates={day}&groups=50&limit={LIMIT}")
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


def parse_event(e):
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

    o = _pick_odds(comp)
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}

    spread = _num(_first(o.get("spread"), _dig(o, "pointSpread", "home", "close", "line")))
    details = str(o.get("details") or "")
    # cross-check sign against 'ABBR -7.5' text; trust the text if they disagree
    parts = details.split()
    if len(parts) == 2 and _num(parts[1]) is not None:
        fav_line = -abs(_num(parts[1]))
        if parts[0] == row["home_abbr"]:
            spread = fav_line
        elif parts[0] == row["away_abbr"]:
            spread = -fav_line
    elif details.upper() in ("EVEN", "PK", "PICK"):
        spread = 0.0

    row.update({
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
    })
    # moneylines between -100 and +100 are not real American odds
    for k in ("ml_home", "ml_away", "spread_home_odds", "spread_away_odds", "over_odds", "under_odds"):
        if row[k] is not None and abs(row[k]) < 100:
            row[k] = None
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

"""Simulated ESPN scoreboard payloads (same JSON shape) so the pipeline can be tested offline.
True team strengths are known, so we can check the model recovers them."""
import datetime as dt
import hashlib

import numpy as np

N = 360
_rng = np.random.default_rng(7)
TRUE = {}
base_o, base_d = _rng.normal(0, 6, N), _rng.normal(0, 5, N)   # per 100 possessions
base_t = _rng.normal(0, 2.5, N)                                # possessions added per game
TEMPO = {}
for s in (2025, 2026, 2027):  # strengths carry over ~60% year to year
    TRUE[s] = (base_o.copy(), base_d.copy())
    TEMPO[s] = base_t.copy()
    base_o = 0.6 * base_o + _rng.normal(0, 4.5, N)
    base_d = 0.6 * base_d + _rng.normal(0, 3.5, N)
    base_t = 0.7 * base_t + _rng.normal(0, 1.8, N)
PPP, POSS = 104.0, 68.0
BOX = {}
MU, HCA, NOISE = 72.0, 3.2, 8.0
TODAY = dt.date.today()


def _season(d):
    return d.year + 1 if d.month >= 7 else d.year


def _ml(spread_home):
    p = 1 / (1 + 10 ** (spread_home / 13.5))  # rough
    def am(q):
        q = min(max(q * 1.045, 0.02), 0.98)  # add vig
        return -round(100 * q / (1 - q)) if q > 0.5 else round(100 * (1 - q) / q)
    return am(p), am(1 - p)


def fake_day(day):
    d = dt.datetime.strptime(day, "%Y%m%d").date()
    s = _season(d)
    if s not in TRUE or not (dt.date(s - 1, 11, 3) <= d <= dt.date(s, 3, 15)):
        return []
    rng = np.random.default_rng(int(hashlib.md5(day.encode()).hexdigest()[:8], 16))
    o, de = TRUE[s]
    n = int(rng.integers(20, 70))
    teams = rng.permutation(N)[: 2 * n]
    events = []
    for k in range(n):
        h, a = int(teams[2 * k]), int(teams[2 * k + 1])
        nonD1 = rng.random() < 0.05
        neutral = rng.random() < 0.1
        adv = 0 if neutral else 4.6 / 2
        aid, ao_, ad_, at_ = (9000 + a, -18.0, 14.0, 0.0) if nonD1 else (a, o[a], de[a], TEMPO[s][a])
        poss = POSS + TEMPO[s][h] + at_
        hp = poss * (PPP + o[h] + ad_ + adv) / 100
        ap = poss * (PPP + ao_ + de[h] - adv) / 100
        poss_act = poss + rng.normal(0, 3.0)
        true_margin = hp - ap
        mkt_margin = true_margin + rng.normal(0, 2.0)
        spread = round(-mkt_margin * 2) / 2
        total = round((hp + ap + rng.normal(0, 3)) * 2) / 2
        tip = dt.datetime.combine(d, dt.time(23, 0)) + dt.timedelta(minutes=int(rng.integers(0, 240)))
        final = d < TODAY
        hs = int(round(hp + rng.normal(0, NOISE)))
        as_ = int(round(ap + rng.normal(0, NOISE)))
        if hs == as_:
            hs += 1
        mlh, mla = _ml(spread)
        if final:
            box = {}
            for side, pts in (("h", hs), ("a", as_)):
                fta, to, oreb = int(rng.integers(12, 30)), int(rng.integers(7, 17)), int(rng.integers(5, 15))
                fga = int(round(poss_act - to - 0.475 * fta + oreb))
                box[side] = {"fga": fga, "fta": fta, "to": to, "oreb": oreb}
        fav_abbr = f"T{h}" if spread < 0 else f"T{aid}"
        odds = [] if rng.random() < 0.08 or nonD1 else [{
            "provider": {"name": "DraftKings"},
            "details": "EVEN" if spread == 0 else f"{fav_abbr} {-abs(spread)}",
            "spread": spread, "overUnder": total,
            "homeTeamOdds": {"favorite": spread < 0, "moneyLine": mlh, "spreadOdds": -110},
            "awayTeamOdds": {"favorite": spread > 0, "moneyLine": mla, "spreadOdds": -110},
            "pointSpread": {"home": {"open": {"line": f"{spread + 1:+.1f}"}}},
            "total": {"over": {"open": {"line": f"o{total - 1}"}}},
        }]
        if final:
            BOX[f"{day}{k:03d}"] = (str(h), str(aid), box, odds)
        events.append({
            "id": f"{day}{k:03d}", "date": tip.strftime("%Y-%m-%dT%H:%MZ"), "season": {"year": s},
            "competitions": [{
                "neutralSite": neutral, "conferenceCompetition": False,
                "status": {"type": {"state": "post" if final else "pre", "completed": final}},
                "competitors": [
                    {"homeAway": "home", "score": str(hs) if final else "0",
                     "team": {"id": str(h), "shortDisplayName": f"Team {h}", "abbreviation": f"T{h}"}},
                    {"homeAway": "away", "score": str(as_) if final else "0",
                     "team": {"id": str(aid), "shortDisplayName": f"Team {aid}", "abbreviation": f"T{aid}"}},
                ],
                "odds": [] if final else odds,  # like ESPN: lines vanish after tip
            }],
        })
    return events


def fake_summary(event_id):
    day = event_id[:8]
    if event_id not in BOX:
        fake_day(day)
    if event_id not in BOX:
        return {}
    hid, aid, box, odds = BOX[event_id]
    def team(tid, b):
        return {"team": {"id": tid}, "statistics": [
            {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": f"{int(b['fga'] * .45)}-{b['fga']}"},
            {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": f"{int(b['fta'] * .7)}-{b['fta']}"},
            {"name": "offensiveRebounds", "displayValue": str(b["oreb"])},
            {"name": "totalTurnovers", "displayValue": str(b["to"])}]}
    return {"pickcenter": odds, "boxscore": {"teams": [team(aid, box["a"]), team(hid, box["h"])]}}

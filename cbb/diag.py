"""Print the shape of ESPN game summaries (what data we can build on)."""
import json
import pandas as pd
from . import espn, store


def shape(x, depth=0, maxd=4):
    pad = "  " * depth
    if isinstance(x, dict):
        for k, v in list(x.items())[:40]:
            if isinstance(v, (dict, list)) and depth < maxd:
                print(f"{pad}{k}: {type(v).__name__}{'['+str(len(v))+']' if isinstance(v, list) else ''}")
                shape(v, depth + 1, maxd)
            else:
                print(f"{pad}{k}: {repr(v)[:80]}")
    elif isinstance(x, list) and x:
        shape(x[0], depth, maxd)


def main():
    g = store.finished(store.load_games())
    picks = [g[g.season == 2026].iloc[3000], g[g.season == 2027].iloc[-50]]
    for r in picks:
        print("=" * 30, r["id"], r["date"], r["home"], "vs", r["away"])
        d = espn._get_json(f"{espn.API}/{store.SPORT.espn}/summary?event={r['id']}")
        print("TOP KEYS:", list(d.keys()))
        print("--- pickcenter"); print(json.dumps(d.get("pickcenter"), indent=1)[:3500])
        print("--- odds"); print(json.dumps(d.get("odds"), indent=1)[:1500])
        print("--- boxscore.teams[0]"); print(json.dumps((d.get("boxscore") or {}).get("teams", [{}])[0], indent=1)[:4000])
        print("--- injuries"); print(json.dumps(d.get("injuries"), indent=1)[:800])
        print("--- predictor"); print(json.dumps(d.get("predictor"), indent=1)[:600])
        print("--- header shape"); shape(d.get("header"), maxd=3)


if __name__ == "__main__":
    main()

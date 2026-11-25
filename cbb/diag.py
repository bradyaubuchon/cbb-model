"""Print the shape of ESPN feeds we may build on."""
import json
from . import espn, store


def show(label, obj, n=2500):
    print(f"\n---------- {label}")
    print(json.dumps(obj, indent=1)[:n])


def main():
    api = espn.API
    for sp in ("basketball/nba", "football/nfl", "basketball/mens-college-basketball"):
        try:
            d = espn._get_json(f"{api}/{sp}/injuries")
            print(f"\n===== {sp} injuries: top keys {list(d.keys())}")
            inj = d.get("injuries") or []
            print("teams:", len(inj))
            if inj:
                show("first team (trimmed)", {**inj[0], "injuries": (inj[0].get("injuries") or [])[:2]}, 3000)
        except Exception as e:
            print(f"\n===== {sp} injuries: ERROR {e}")
    store.set_sport("nba")
    g = store.finished(store.load_games())
    gid = g.iloc[-5]["id"]
    d = espn._get_json(f"{api}/basketball/nba/summary?event={gid}")
    pl = (d.get("boxscore") or {}).get("players") or []
    print("\n===== NBA summary players sections:", len(pl))
    if pl:
        t = pl[0]
        st = (t.get("statistics") or [{}])[0]
        show("team keys", list(t.keys()))
        show("stat block keys/labels", {k: v for k, v in st.items() if k != "athletes"})
        show("first 2 athletes", (st.get("athletes") or [])[:2], 2500)
    show("injuries in summary", d.get("injuries"), 1500)


if __name__ == "__main__":
    main()

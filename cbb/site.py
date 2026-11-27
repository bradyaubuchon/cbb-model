"""Builds the phone page in docs/ (served by GitHub Pages).

docs/index.html   the app (changes only when the app's code changes)
docs/data.json    picks, results, backtest, ratings (changes every run)
docs/version.json {app, data} - the page polls this to offer "Tap to update"
"""
import hashlib
import json
import math
import shutil

from . import store

DOCS = store.ROOT / "docs"
WEB = store.ROOT / "cbb" / "web"
APP_RELEASE = "4.2"  # bump for notable app changes; the build hash changes automatically


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def app_version():
    h = hashlib.sha256()
    for f in sorted(WEB.iterdir()):
        h.update(f.read_bytes())
    return f"{APP_RELEASE}-{h.hexdigest()[:7]}"


def render(sport, slate, ratings, rec, bt, p, now, calibrated=True, odds_time=None):
    """Write docs/data-<sport>.json. Returns its content hash (the data version)."""
    tab = ratings.table().head(400)
    data = {
        "sport": sport,
        "updated": now.strftime("%a %b %-d, %-I:%M %p ET"),
        "updated_iso": now.isoformat(),
        "params": p.to_dict(),
        "calibrated": calibrated,
        "odds_time": odds_time,
        "odds_min_ev": 0.02,
        "home_word": store.SPORT.home_word,
        "horizon_h": store.SPORT.horizon_h,
        "slate": slate,
        "record": rec,
        "backtest": bt,
        "hca": ratings.hca, "mu": ratings.mu, "rest_w": getattr(ratings, "rest_w", 0.0),
        "ratings": tab[[c for c in ("team", "off", "def", "net", "games", "adj_o", "adj_d", "adj_em", "tempo")
                        if c in tab.columns]].round(2).to_dict("records"),
    }
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(_clean(data), default=float, separators=(",", ":"))
    (DOCS / f"data-{sport}.json").write_text(payload)
    if sport == "ncaam":  # app v2.0 (single sport) reads data.json; keep it working until it updates
        (DOCS / "data.json").write_text(payload)
    return {"v": hashlib.sha256(payload.encode()).hexdigest()[:10], "updated": data["updated"],
            "iso": data["updated_iso"]}


def write_version(versions):
    """Copy the app files and write version.json, which open apps poll for 'Tap to update'."""
    DOCS.mkdir(exist_ok=True)
    old = {}
    vf = DOCS / "version.json"
    if vf.exists():
        try:
            old = json.loads(vf.read_text()).get("sports", {})
        except Exception:
            old = {}
    sports = {**old, **{k: v for k, v in versions.items() if v}}  # a failed sport keeps its last data
    app = app_version()
    for f in WEB.iterdir():
        if f.name == "index.html":
            (DOCS / "index.html").write_text(f.read_text().replace("__APP_VERSION__", app))
        else:
            shutil.copy(f, DOCS / f.name)
    latest = max(sports.values(), key=lambda v: v.get("iso", ""), default={}).get("updated", "")
    (DOCS / "version.json").write_text(json.dumps({
        "app": app,
        "data": hashlib.sha256(json.dumps(sports, sort_keys=True).encode()).hexdigest()[:10],
        "updated": latest,
        "sports": sports,
    }))
    (DOCS / ".nojekyll").write_text("")

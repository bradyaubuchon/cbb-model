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
APP_RELEASE = "2.0"  # bump for notable app changes; the build hash changes automatically


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


def render(slate, ratings, rec, bt, p, now, calibrated=True):
    tab = ratings.table().head(400)
    data = {
        "updated": now.strftime("%a %b %-d, %-I:%M %p ET"),
        "updated_iso": now.isoformat(),
        "params": p.to_dict(),
        "calibrated": calibrated,
        "slate": slate,
        "record": rec,
        "backtest": bt,
        "hca": ratings.hca, "mu": ratings.mu,
        "ratings": tab[[c for c in ("team", "off", "def", "net", "games", "adj_o", "adj_d", "adj_em", "tempo")
                        if c in tab.columns]].round(2).to_dict("records"),
    }
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(_clean(data), default=float, separators=(",", ":"))
    data_ver = hashlib.sha256(payload.encode()).hexdigest()[:10]
    app = app_version()
    (DOCS / "data.json").write_text(payload)
    for f in WEB.iterdir():
        if f.name == "index.html":
            (DOCS / "index.html").write_text(f.read_text().replace("__APP_VERSION__", app))
        else:
            shutil.copy(f, DOCS / f.name)
    (DOCS / "version.json").write_text(json.dumps({"app": app, "data": data_ver, "updated": data["updated"]}))
    (DOCS / ".nojekyll").write_text("")

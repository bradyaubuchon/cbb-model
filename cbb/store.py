"""CSV storage for games and tracked picks."""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .sports import SPORTS

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
SPORT = SPORTS["ncaam"]
DATA = GAMES = PICKS = PARAMS = BACKTEST = None


def set_sport(key):
    """Point every path at data/<sport>/. The pipeline runs one sport at a time."""
    global SPORT, DATA, GAMES, PICKS, PARAMS, BACKTEST
    SPORT = SPORTS[key]
    DATA = ROOT / "data" / key
    GAMES, PICKS = DATA / "games.csv", DATA / "picks.json"
    PARAMS, BACKTEST = DATA / "params.json", DATA / "backtest.json"
    DATA.mkdir(parents=True, exist_ok=True)


def migrate_legacy():
    """Before multi-sport, college data lived directly in data/. Move it to data/ncaam/."""
    old = ROOT / "data"
    files = [f for f in old.iterdir() if f.is_file()] if old.exists() else []
    if files:
        dest = old / "ncaam"
        dest.mkdir(exist_ok=True)
        for f in files:
            f.rename(dest / f.name)

ODDS_COLS = ["book", "spread", "spread_open", "spread_home_odds", "spread_away_odds", "total",
             "total_open", "over_odds", "under_odds", "ml_home", "ml_away"]


def season_of(day):
    return SPORT.season_of(day)


def season_days(season):
    return SPORT.season_days(season)


def load_games():
    if not GAMES.exists():
        return pd.DataFrame()
    df = pd.read_csv(GAMES, dtype={"id": str, "home_id": str, "away_id": str})
    return df


def upsert_games(old, rows):
    """Newer rows replace older ones by id, but a finished game keeps the last odds we saw
    before tip if ESPN has since dropped them."""
    new = pd.DataFrame(rows)
    if new.empty:
        return old
    if old.empty:
        out = new
    else:
        prev = old.set_index("id")
        new = new.set_index("id")
        both = new.index.intersection(prev.index)
        # keep anything we already know (pre-tip odds, box scores) that the fresh pull lacks
        for col in prev.columns:
            if col not in new.columns:
                new[col] = float("nan") if prev[col].dtype.kind in "fi" else None
            miss = new.loc[both, col].isna()
            new.loc[both[miss.values], col] = prev.loc[both[miss.values], col]
        out = pd.concat([prev.drop(index=both), new]).reset_index()
    out = out.sort_values(["date", "id"]).reset_index(drop=True)
    DATA.mkdir(exist_ok=True)
    out.to_csv(GAMES, index=False)
    return out


def finished(df):
    if df.empty:
        return df
    return df[(df["state"] == "post") & df["home_score"].notna() & df["away_score"].notna()]


def now_et():
    return dt.datetime.now(ET)


def to_et_date(iso_series):
    return pd.to_datetime(iso_series, utc=True).dt.tz_convert(ET).dt.date

"""CSV storage for games and tracked picks."""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GAMES = DATA / "games.csv"
PICKS = DATA / "picks.json"
PARAMS = DATA / "params.json"
BACKTEST = DATA / "backtest.json"
ET = ZoneInfo("America/New_York")

ODDS_COLS = ["book", "spread", "spread_open", "spread_home_odds", "spread_away_odds", "total",
             "total_open", "over_odds", "under_odds", "ml_home", "ml_away"]


def season_of(day):
    """ESPN season year: the 2026-27 season is 2027."""
    return day.year + 1 if day.month >= 7 else day.year


def season_days(season):
    """Every ET calendar day that can hold a D-I game (Nov 1 through the title game)."""
    d, end = dt.date(season - 1, 11, 1), dt.date(season, 4, 10)
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


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

"""Per-sport settings. Every sport runs the same pipeline; these are the differences."""
import datetime as dt
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Sport:
    key: str               # folder / URL name
    name: str              # shown in the app
    espn: str              # ESPN API path
    query: str             # extra scoreboard query
    efficiency: bool       # box scores give possessions (basketball only)
    min_lines_to_tune: int  # last-season games with closing lines needed to fit the market blend
    season_of: Callable    # date -> ESPN season year
    season_days: Callable  # season year -> iterable of dates that can hold games
    home_word: str         # "home court" / "home field"
    horizon_h: int = 40    # how far ahead the slate looks
    odds_key: str = ""     # The Odds API sport key
    rest_cap: tuple = (1, 3, 1)  # days of rest counted (min, max), and scale of the rest feature


def _days(start, end):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


SPORTS = {
    "ncaam": Sport(
        "ncaam", "College basketball", "basketball/mens-college-basketball", "&groups=50", True, 3000,
        lambda d: d.year + 1 if d.month >= 7 else d.year,                      # 2026-27 season = 2027
        lambda s: _days(dt.date(s - 1, 11, 1), dt.date(s, 4, 10)), "home court", 40, "basketball_ncaab"),
    "nba": Sport(
        "nba", "NBA", "basketball/nba", "", True, 800,
        lambda d: d.year + 1 if d.month >= 8 else d.year,                      # 2026-27 season = 2027
        lambda s: _days(dt.date(s - 1, 10, 15), dt.date(s, 6, 25)), "home court", 40, "basketball_nba"),
    "nfl": Sport(
        "nfl", "NFL", "football/nfl", "", False, 200,
        lambda d: d.year if d.month >= 3 else d.year - 1,                      # 2026-27 season = 2026
        lambda s: _days(dt.date(s, 9, 1), dt.date(s + 1, 2, 15)), "home field", 168, "americanfootball_nfl", (4, 14, 7)),
}
ORDER = ["ncaam", "nba", "nfl"]

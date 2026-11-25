# Betting Model

A personal betting model for men's college basketball, the NBA and the NFL. It runs on GitHub Actions every 2 hours, at no cost. Each run:

1. Pulls ESPN's public D-I scoreboard, which carries scores plus the spread, total and moneyline ESPN shows (DraftKings).
2. Fits offense and defense ratings for every team with weighted ridge regression on points scored. Recent games count more, and last season's games seed the early-season ratings.
3. Blends the ratings with the market line. A backtest picks how much weight the ratings get, since the line is a strong forecast by itself.
4. Prices the spread, total and moneyline for every game in the next 40 hours and flags plays at EV ≥ 3%.
5. Logs each pick at the first line it saw and grades it after the game, including closing line value.
6. Publishes `docs/index.html`, which GitHub Pages serves as your phone page.

Every Monday it re-tunes the rating knobs and the blend weights on last season, then reruns the walk-forward backtest.

## Honest caveats
- **One book only.** ESPN shows a single sportsbook's line, so there's no line shopping. The "first seen" line is whatever ESPN showed when the job first ran, not the true opener.
- **Points only.** No tempo, injuries, lineups or travel. That's a reasonable first model, but sharp books already price all of it.
- **Judge it by CLV and the out-of-sample backtest**, not by a hot week.

## Running locally
```
pip install -r requirements.txt
python -m cbb.run update      # all sports: pull data, price slate, write site
python -m cbb.run update nba  # one sport
python -m cbb.run tune        # re-fit parameters + backtest
CBB_FAKE=1 python -m cbb.run update   # offline test on simulated data
```

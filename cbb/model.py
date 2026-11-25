"""Team ratings and game pricing.

Every team gets an offense rating o and a defense rating d (points above a D-I average
team, d > 0 means a leaky defense). For each finished game, two equations:

    home_pts = mu + o_home + d_away + hca/2   (hca term is 0 on neutral floors)
    away_pts = mu + o_away + d_home - hca/2

Solved by weighted ridge regression. Weights decay with game age; last season's games enter
at a reduced weight so early-season ratings start from last year instead of from zero.
Ridge shrinks every team toward average, which matters when a team has played 4 games.
"""
import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

POOL = "POOL"  # teams with very few D-I games (mostly non-D-I opponents) share one rating


@dataclass
class Params:
    half_life: float = 60.0      # days for a current-season game's weight to halve
    lam: float = 10.0            # ridge strength (games-equivalent of shrinkage to average)
    prev_w: float = 0.35         # weight of last season's games
    pool_min: int = 6            # teams with fewer games than this are pooled
    margin_sd: float = 11.0      # sd of (actual margin - model margin); set by tuning
    total_sd: float = 17.5       # sd of (actual total - model total); set by tuning
    min_ev: float = 0.03         # EV per unit needed to flag a play
    ml_flag: bool = False        # backtest: flagged moneylines lost ~7%/bet, so they're shown but not flagged
    w_margin: float = 0.3        # weight on the model vs the market line for the margin (tuned)
    w_total: float = 0.3         # same for the total (tuned)
    # efficiency + pace model (per-100-possession ratings, KenPom-style)
    e_half_life: float = 60.0
    e_lam: float = 8.0
    e_prev_w: float = 0.35
    t_lam: float = 5.0           # ridge strength for tempo
    mix_margin: float = 0.0      # share of the efficiency model in the ratings forecast (tuned)
    mix_total: float = 0.0

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})

    def to_dict(self):
        return asdict(self)


def pooled_ids(games, pool_min):
    counts = pd.concat([games["home_id"], games["away_id"]]).value_counts()
    return set(counts[counts < pool_min].index)


class Ratings:
    def __init__(self, mu, hca, off, dfn, names, n_games):
        self.mu, self.hca, self.off, self.dfn = mu, hca, off, dfn
        self.names, self.n_games = names, n_games

    def _get(self, tid, pool):
        tid = POOL if tid in pool or tid not in self.off else tid
        return self.off.get(tid, 0.0), self.dfn.get(tid, 0.0)

    def predict(self, home_id, away_id, neutral, pool=()):
        oh, dh = self._get(home_id, pool)
        oa, da = self._get(away_id, pool)
        adv = 0.0 if neutral else self.hca / 2
        hp = self.mu + oh + da + adv
        ap = self.mu + oa + dh - adv
        return hp - ap, hp + ap, hp, ap

    def table(self):
        rows = [{"id": t, "team": self.names.get(t, t), "off": self.off[t], "def": self.dfn[t],
                 "net": self.off[t] - self.dfn[t], "games": self.n_games.get(t, 0)}
                for t in self.off if t != POOL]
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def fit(done, as_of, season, p, pool):
    """done: finished games strictly before as_of (a pandas Timestamp, UTC)."""
    g = done[done["season"] >= season - 1]
    if g.empty:
        return None
    t = pd.to_datetime(g["date"], utc=True)
    cur = (g["season"] == season).to_numpy()
    age = (as_of - t).dt.total_seconds().to_numpy() / 86400
    prev_end = t[~cur].max() if (~cur).any() else None
    w = np.where(cur, 0.5 ** (age / p.half_life), 0.0)
    if prev_end is not None:
        prev_age = ((prev_end - t).dt.total_seconds().to_numpy() / 86400)
        w = np.where(cur, w, p.prev_w * 0.5 ** (prev_age / p.half_life))

    hid = g["home_id"].where(~g["home_id"].isin(pool), POOL).to_numpy()
    aid = g["away_id"].where(~g["away_id"].isin(pool), POOL).to_numpy()
    teams = sorted(set(hid) | set(aid))
    ix = {tm: i for i, tm in enumerate(teams)}
    n, m = len(teams), len(g)
    hi = np.array([ix[x] for x in hid])
    ai = np.array([ix[x] for x in aid])
    adv = np.where(g["neutral"].to_numpy() == 1, 0.0, 0.5)

    # columns: 0 mu, 1 hca, 2..n+1 offense, n+2..2n+1 defense
    r = np.arange(2 * m)
    rows = np.concatenate([r, r, r, r])
    cols = np.concatenate([
        np.zeros(2 * m, int), np.ones(2 * m, int),
        np.concatenate([2 + hi, 2 + ai]),
        np.concatenate([2 + n + ai, 2 + n + hi]),
    ])
    vals = np.concatenate([np.ones(2 * m), np.concatenate([adv, -adv]), np.ones(2 * m), np.ones(2 * m)])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 * n + 2))
    y = np.concatenate([g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)])
    W = np.concatenate([w, w])

    XtW = X.T.multiply(W)
    A = (XtW @ X).tocsc()
    pen = np.full(2 * n + 2, p.lam)
    pen[:2] = 1e-6
    A = A.tolil()
    A.setdiag(A.diagonal() + pen)
    A = A.tocsc()
    beta = spsolve(A, XtW @ y)

    names = {}
    for idcol, ncol in (("home_id", "home"), ("away_id", "away")):
        names.update(dict(zip(g[idcol], g[ncol])))
    cnt = pd.concat([g.loc[cur, "home_id"], g.loc[cur, "away_id"]]).value_counts().to_dict()
    off = {tm: beta[2 + i] for tm, i in ix.items()}
    dfn = {tm: beta[2 + n + i] for tm, i in ix.items()}
    return Ratings(beta[0], beta[1], off, dfn, names, cnt)


# ---------- pricing ----------

def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob_over(mean, sd, line):
    """P(X > line), P(X == line) for an integer-valued score X ~ Normal(mean, sd)."""
    if float(line).is_integer():
        push = _phi((line + 0.5 - mean) / sd) - _phi((line - 0.5 - mean) / sd)
        win = 1 - _phi((line + 0.5 - mean) / sd)
    else:
        push = 0.0
        win = 1 - _phi((math.floor(line) + 0.5 - mean) / sd)
    return win, push


def payout(american):
    return american / 100 if american > 0 else 100 / -american


def ev(p_win, p_push, american):
    return p_win * payout(american) - (1 - p_win - p_push)


def implied(american):
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def blend(margin, total, g, p):
    """Shrink the model toward the market. The market line is a strong forecast on its own;
    the tuned weight says how much the model adds on top of it (0 = nothing)."""
    if _ok(g.get("spread")):
        margin = p.w_margin * margin + (1 - p.w_margin) * -g["spread"]
    if _ok(g.get("total")):
        total = p.w_total * total + (1 - p.w_total) * g["total"]
    return margin, total


def price_game(margin, total, g, p):
    """All three markets for one game. g: row with ESPN lines. Returns dict of bets."""
    out = {}
    margin, total = blend(margin, total, g, p)
    L = g.get("spread")
    if _ok(L):
        hw, push = prob_over(margin, p.margin_sd, -L)
        aw = 1 - hw - push
        ho = g.get("spread_home_odds") if _ok(g.get("spread_home_odds")) else -110
        ao = g.get("spread_away_odds") if _ok(g.get("spread_away_odds")) else -110
        side = [("home", hw, ho), ("away", aw, ao)]
        best = max(side, key=lambda s: ev(s[1], push, s[2]))
        out["spread"] = {"side": best[0], "p": best[1], "odds": best[2], "ev": ev(best[1], push, best[2]),
                         "push": push, "line": L if best[0] == "home" else -L}
    T = g.get("total")
    if _ok(T):
        ow, push = prob_over(total, p.total_sd, T)
        uw = 1 - ow - push
        oo = g.get("over_odds") if _ok(g.get("over_odds")) else -110
        uo = g.get("under_odds") if _ok(g.get("under_odds")) else -110
        side = [("over", ow, oo), ("under", uw, uo)]
        best = max(side, key=lambda s: ev(s[1], push, s[2]))
        out["total"] = {"side": best[0], "p": best[1], "odds": best[2], "ev": ev(best[1], push, best[2]),
                        "push": push, "line": T}
    mh, ma = g.get("ml_home"), g.get("ml_away")
    if _ok(mh) and _ok(ma):
        w, tie = prob_over(margin, p.margin_sd, 0)
        ph = w / (1 - tie)  # no ties in basketball: overtime splits the 0-margin mass
        side = [("home", ph, mh), ("away", 1 - ph, ma)]
        best = max(side, key=lambda s: ev(s[1], 0, s[2]))
        ih, ia = implied(mh), implied(ma)
        if best[2] > 400:  # normal-curve tails are unreliable for big underdogs; don't price them
            return out
        mkt = ih / (ih + ia) if best[0] == "home" else ia / (ih + ia)
        out["ml"] = {"side": best[0], "p": best[1], "odds": best[2], "ev": ev(best[1], 0, best[2]),
                     "market_p": mkt}
    return out


def grade(bet, market, hs, as_):
    """Profit in units (risking 1) for a priced bet given the final score, or None for a push."""
    margin, tot = hs - as_, hs + as_
    if market == "spread":
        m = margin if bet["side"] == "home" else -margin
        res = m + bet["line"]
    elif market == "total":
        res = (tot - bet["line"]) * (1 if bet["side"] == "over" else -1)
    else:
        res = margin if bet["side"] == "home" else -margin
    if res == 0:
        return 0.0
    return payout(bet["odds"]) if res > 0 else -1.0


# ---------- efficiency + pace model ----------

def possessions(g):
    """Possessions per team per game, averaged over both teams (standard estimate)."""
    ph = g["h_fga"] - g["h_oreb"] + g["h_to"] + 0.475 * g["h_fta"]
    pa = g["a_fga"] - g["a_oreb"] + g["a_to"] + 0.475 * g["a_fta"]
    return (ph + pa) / 2


def _weights(g, as_of, season, half_life, prev_w):
    t = pd.to_datetime(g["date"], utc=True)
    cur = (g["season"] == season).to_numpy()
    age = (as_of - t).dt.total_seconds().to_numpy() / 86400
    w = np.where(cur, 0.5 ** (age / half_life), 0.0)
    if (~cur).any():
        prev_end = t[~cur].max()
        prev_age = (prev_end - t).dt.total_seconds().to_numpy() / 86400
        w = np.where(cur, w, prev_w * 0.5 ** (prev_age / half_life))
    return w, cur


def _index(g, pool):
    hid = g["home_id"].where(~g["home_id"].isin(pool), POOL).to_numpy()
    aid = g["away_id"].where(~g["away_id"].isin(pool), POOL).to_numpy()
    teams = sorted(set(hid) | set(aid))
    ix = {tm: i for i, tm in enumerate(teams)}
    return ix, np.array([ix[x] for x in hid]), np.array([ix[x] for x in aid])


def _solve(X, y, W, lam, free):
    XtW = X.T.multiply(W)
    A = (XtW @ X).tolil()
    pen = np.full(X.shape[1], lam)
    pen[:free] = 1e-6
    A.setdiag(A.diagonal() + pen)
    return spsolve(A.tocsc(), XtW @ y)


class EffRatings:
    """o/d in points per 100 possessions vs average; t = possessions added to the game."""

    def __init__(self, mu, hca, off, dfn, mu_t, tempo):
        self.mu, self.hca, self.off, self.dfn, self.mu_t, self.tempo = mu, hca, off, dfn, mu_t, tempo

    def predict(self, home_id, away_id, neutral, pool=()):
        def k(t):
            return POOL if t in pool or t not in self.off else t
        h, a = k(home_id), k(away_id)
        poss = self.mu_t + self.tempo.get(h, 0.0) + self.tempo.get(a, 0.0)
        adv = 0.0 if neutral else self.hca / 2
        hp = poss * (self.mu + self.off.get(h, 0) + self.dfn.get(a, 0) + adv) / 100
        ap = poss * (self.mu + self.off.get(a, 0) + self.dfn.get(h, 0) - adv) / 100
        return hp - ap, hp + ap, hp, ap


def fit_eff(done, as_of, season, p, pool):
    g = done[(done["season"] >= season - 1) & done["h_fga"].notna() & done["a_fga"].notna()] \
        if "h_fga" in done.columns else done.iloc[0:0]
    if len(g) < 500:
        return None
    poss = possessions(g).to_numpy(float)
    ok = (poss > 45) & (poss < 110)
    g, poss = g[ok], poss[ok]
    w, _ = _weights(g, as_of, season, p.e_half_life, p.e_prev_w)
    ix, hi, ai = _index(g, pool)
    n, m = len(ix), len(g)
    adv = np.where(g["neutral"].to_numpy() == 1, 0.0, 0.5)

    r = np.arange(2 * m)
    X = sp.csr_matrix((
        np.concatenate([np.ones(2 * m), np.concatenate([adv, -adv]), np.ones(2 * m), np.ones(2 * m)]),
        (np.concatenate([r, r, r, r]),
         np.concatenate([np.zeros(2 * m, int), np.ones(2 * m, int),
                         np.concatenate([2 + hi, 2 + ai]), np.concatenate([2 + n + ai, 2 + n + hi])]))),
        shape=(2 * m, 2 * n + 2))
    y = 100 * np.concatenate([g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)]) / np.concatenate([poss, poss])
    wp = w * poss / 70  # more possessions = more information about efficiency
    b = _solve(X, y, np.concatenate([wp, wp]), p.e_lam, 2)

    rt = np.arange(m)
    Xt = sp.csr_matrix((np.ones(3 * m), (np.concatenate([rt, rt, rt]),
                                         np.concatenate([np.zeros(m, int), 1 + hi, 1 + ai]))), shape=(m, n + 1))
    bt = _solve(Xt, poss, w, p.t_lam, 1)
    return EffRatings(b[0], b[1], {t: b[2 + i] for t, i in ix.items()}, {t: b[2 + n + i] for t, i in ix.items()},
                      bt[0], {t: bt[1 + i] for t, i in ix.items()})


class Combo:
    """Ratings forecast = mix of the points model and the efficiency model (tuned shares)."""

    def __init__(self, pts, eff, p):
        self.pts, self.eff, self.p = pts, eff, p
        self.n_games, self.hca, self.mu = pts.n_games, pts.hca, pts.mu

    def predict(self, home_id, away_id, neutral, pool=()):
        m1, t1, _, _ = self.pts.predict(home_id, away_id, neutral, pool)
        if self.eff is None:
            m, t = m1, t1
        else:
            m2, t2, _, _ = self.eff.predict(home_id, away_id, neutral, pool)
            m = (1 - self.p.mix_margin) * m1 + self.p.mix_margin * m2
            t = (1 - self.p.mix_total) * t1 + self.p.mix_total * t2
        return m, t, (t + m) / 2, (t - m) / 2

    def table(self):
        tab = self.pts.table()
        if self.eff is not None:
            e = self.eff
            tab["adj_o"] = tab["id"].map(lambda t: e.mu + e.off.get(t, 0))
            tab["adj_d"] = tab["id"].map(lambda t: e.mu + e.dfn.get(t, 0))
            tab["tempo"] = tab["id"].map(lambda t: e.mu_t + 2 * e.tempo.get(t, 0))  # vs an average-pace opponent
            tab["adj_em"] = tab["adj_o"] - tab["adj_d"]
        return tab


def fit_all(done, as_of, season, p, pool):
    pts = fit(done, as_of, season, p, pool)
    if pts is None:
        return None
    return Combo(pts, fit_eff(done, as_of, season, p, pool), p)

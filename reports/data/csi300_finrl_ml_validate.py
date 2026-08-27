"""Validation for the FinRL-ML quarterly sleeve (2026-08-21).

E4 (quarterly top-25, Sharpe 1.15) beat stable-7's 1.04 — but the monthly
track taught us to check two things before believing a long-horizon sleeve:
  1. ex-2020 Sharpe (2020 was a one-off alpha spike)
  2. daily-return correlation vs stable-7 (diversification or clone?)
Plus yearly breakdown and the 50/50 portfolio combo.
Loads the cached score from csi300_finrl_ml_test.py (finrl_ml_score.pkl).
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, volume = panel["close"], panel["volume"]
days = close.index
fwd1 = close.pct_change().shift(-1)
ret = close.pct_change()
score = pickle.load(open(DATA / "finrl_ml_score.pkl", "rb"))

TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

tradable = close.notna() & volume.fillna(0).gt(0)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd1_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

def cz(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

def constrained_net(sig: pd.DataFrame, top_n: int = TOP_N, rebal: int = REBAL) -> pd.Series:
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % rebal == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= top_n:
                desired = set(rowv.nlargest(top_n).index)
                locked, keep = set(), held & desired
                for s in held - desired:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= top_n: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), top_n)
    gross = (w * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

# stable-7 signal (verbatim)
from src.factors.registry import get_default_registry
reg = get_default_registry()
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
fac7 = {a: cz(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac7[a].loc[t].corr(fwd1_clean.loc[t]) for t in win], index=win))
           for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    sig7.loc[days[start:start + RETRAIN]] = sum(fac7[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)

net7 = constrained_net(sig7)
netE4 = constrained_net(score, top_n=25, rebal=63)
combo = (net7 + netE4) / 2

def metrics(net: pd.Series, label: str) -> dict:
    net = net[net.index >= OOS_START]
    eq = (1 + net).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net.std() * np.sqrt(252))
    mdd = float(((eq / eq.cummax()) - 1).min())
    def sharpe(n):
        eqn = (1 + n).cumprod()
        yrs = max((eqn.index[-1] - eqn.index[0]).days / 365.25, 1e-9)
        c = float(eqn.iloc[-1] ** (1 / yrs) - 1)
        return round(c / (n.std() * np.sqrt(252)), 2) if n.std() else None
    ex20 = net[net.index.year != 2020]
    return {"label": label,
            "cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2) if vol else None,
            "sharpe_ex2020": sharpe(ex20),
            "max_dd_pct": round(mdd * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                                eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

corr = float(net7.corr(netE4))
out = {"corr_stable7_vs_E4_daily_returns": round(corr, 3),
       "metrics": [metrics(net7, "stable7"), metrics(netE4, "E4_quarterly_top25"),
                   metrics(combo, "combo_50_50")]}
json.dump(out, open(DATA / "csi300_finrl_ml_validate.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

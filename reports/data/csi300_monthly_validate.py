"""Validation for monthly-4 as second strategy (user chose B, 2026-08-20).

Checks: (1) monthly_equal without 2020 (is it a one-year lottery?);
(2) correlation of daily returns vs stable-7; (3) 50/50 combined Sharpe —
the actual case for a second track.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for k, v in fund.items():
    if k.startswith("fund:"):
        panel[k] = v
close = panel["close"]; volume = panel["volume"]
days = close.index
ret = close.pct_change()
fwd1 = ret.shift(-1)
fwd20 = close.pct_change(20).shift(-20)

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
fwd_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()
print("computing factors...", file=sys.stderr, flush=True)
# monthly-4 factors
mf = {}
for a in ["lit_dnbeta120", "academic_carhart_mom", "fund_earnings_yield"]:
    mf[a] = reg.compute(a, panel)
mf["raw_mom252"] = close.pct_change(252)
for a in mf:
    mu, sd = mf[a].mean(axis=1), mf[a].std(axis=1)
    mf[a] = mf[a].sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
sig_m = sum(mf.values()) / 4
# stable-7 factors + IR blend
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
fac = {a: None for a in STABLE7}
for a in STABLE7:
    f = reg.compute(a, panel).rolling(10, min_periods=6).mean()
    mu, sd = f.mean(axis=1), f.std(axis=1)
    fac[a] = f.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
sig_s = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(252, len(days), 63):
    win = days[start - 252:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + 63]
    sig_s.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())

def backtest(sig, top_n, rebal):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % rebal == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= top_n:
                dset = set(rowv.nlargest(top_n).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= top_n: break
                    if s in held: continue
                    if not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), top_n)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * 0.001

net_m = backtest(sig_m, 30, 20)
net_s = backtest(sig_s, 15, 5)

def stats(net, start=OOS_START, end=None):
    n = net[net.index >= start]
    if end is not None:
        n = n[n.index < end]
    eq = (1 + n).cumprod()
    yrs = max((n.index[-1] - n.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(n.std() * np.sqrt(252))
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2),
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1)}

out = {}
out["monthly4_full"] = stats(net_m)
out["monthly4_ex2020"] = stats(net_m[(net_m.index < "2020-01-01") | (net_m.index >= "2021-01-01")])
out["stable7_full"] = stats(net_s)
oos_m = net_m[net_m.index >= OOS_START]
oos_s = net_s[net_s.index >= OOS_START]
corr = float(oos_m.corr(oos_s))
out["corr_daily_returns"] = round(corr, 3)
combo = (oos_m + oos_s) / 2
eq = (1 + combo).cumprod()
yrs = max((combo.index[-1] - combo.index[0]).days / 365.25, 1e-9)
cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
out["combo_50_50"] = {"cagr_pct": round(cagr * 100, 1),
                      "sharpe": round(cagr / float(combo.std() * np.sqrt(252)), 2),
                      "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1)}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(DATA / "csi300_monthly_validate.json", "w"), ensure_ascii=False, indent=1)
print("SAVED csi300_monthly_validate.json", file=sys.stderr)

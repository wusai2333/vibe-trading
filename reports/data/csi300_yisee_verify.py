"""Yisee multifactor_strategy verification (2026-08-21).

Their recipe: growth(0.33) + momentum(0.34) + quality(0.33), MAD+zscore,
monthly top-10 EW on CSI300 2021-23. Their own numbers: absolute Sharpe
0.068, excess +21.3%/yr vs a crashing benchmark. Verify the factor recipes
on our clean panel + gates:

  momentum = mean(P(t-5)/P(t-25)-1, P(t-5)/P(t-65)-1)   [verbatim]
  quality  = mean(z(fund:roe), z(fund:gross_profitability))
  growth   = z(net_income YoY)  [revenue YoY unavailable in our cache]
  composite= 0.33g + 0.34m + 0.33q (their weights)

Tests: daily IC, gate-2 corr vs stable-7, standalone constrained sleeves
(our top-15/5d mechanism AND their monthly top-10 mechanism).
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
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
fwd_clean = fwd.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

# ---- their factor recipes ----
mom = zscore(((close.shift(5) / close.shift(25) - 1) + (close.shift(5) / close.shift(65) - 1)) / 2)
qual = zscore((zscore(panel["fund:roe"]) + zscore(panel["fund:gross_profitability"])) / 2)
ni = panel["fund:net_income"]
grow = zscore(ni / ni.shift(252) - 1)
comp = zscore(0.33 * grow + 0.34 * mom + 0.33 * qual)
ys = {"momentum": mom, "quality": qual, "growth": grow, "composite": comp}

oos_days = days[days >= OOS_START]
def daily_ic(f):
    ic = pd.Series({t: f.loc[t].corr(fwd_clean.loc[t], method="spearman") for t in oos_days}).dropna()
    return {"ic_mean": round(float(ic.mean()), 4), "ic_ir": round(float(ic.mean() / ic.std()), 3) if ic.std() else None}
print("daily IC...", file=sys.stderr, flush=True)
ics = {k: daily_ic(v) for k, v in ys.items()}

# gate-2 corr of composite vs stable-7
fac7 = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
corrs = {a: round(float(comp.rank(axis=1).corrwith(fac7[a].rank(axis=1), axis=1)[days >= OOS_START].mean()), 3)
         for a in STABLE7}

def net_of(sig, top_n=TOP_N, rebal=REBAL):
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
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), top_n)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

def stats(net, label):
    eq = (1 + net).cumprod(); eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

print("sleeves...", file=sys.stderr, flush=True)
sleeves = {"comp_top15_5d": stats(net_of(comp), "composite our-mechanism"),
           "comp_monthly_top10": stats(net_of(comp, top_n=10, rebal=21), "composite their-mechanism")}
out = {"A_daily_ic": ics, "B_composite_corr_vs_stable7": corrs, "C_sleeves": sleeves}
json.dump(out, open(DATA / "csi300_yisee_verify.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

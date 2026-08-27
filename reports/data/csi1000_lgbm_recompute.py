"""Recompute LGBM-monthly metrics (yearly bug fix) + correlation vs stable-7.

Loads saved pred/monthly from csi1000_lgbm_monthly.pkl — no LGBM rerun.
Adds: proper within-year returns, ex-bull(2024-25) Sharpe, and correlation of
the monthly sleeve vs stable-7 daily net (resampled to monthly).
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
art = pickle.load(open(DATA / "csi1000_lgbm_monthly.pkl", "rb"))
monthly = art["monthly"]

COST_M = 0.001
pred = art["pred"]
N_GROUPS = 20
long_book = {d: set(s.loc[s.group == N_GROUPS - 1, "stock"]) for d, s in pred.groupby("date")}
short_book = {d: set(s.loc[s.group == 0, "stock"]) for d, s in pred.groupby("date")}
ds_ = sorted(long_book)
to_l = [1 - len(long_book[ds_[k]] & long_book[ds_[k-1]]) / max(len(long_book[ds_[k]]), 1)
        for k in range(1, len(ds_))]
to_s = [1 - len(short_book[ds_[k]] & short_book[ds_[k-1]]) / max(len(short_book[ds_[k]]), 1)
        for k in range(1, len(ds_))]
monthly["long_net"] = monthly["long"] - pd.Series([np.nan] + to_l, index=monthly.index) * 2 * COST_M
monthly["ls_net"] = monthly["ls"] - (pd.Series([np.nan] + to_l, index=monthly.index) +
                                       pd.Series([np.nan] + to_s, index=monthly.index)) * 2 * COST_M

def metrics(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    n = len(r)
    cum = (1 + r).prod()
    ann = float(cum ** (12 / n) - 1)
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    exb = r[~r.index.year.isin([2024, 2025])]
    def yearly(eq_):
        out = {}
        for y in sorted(set(eq_.index.year)):
            seg = eq_[eq_.index.year == y]
            out[str(y)] = round(float(seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1)
        return out
    return {"label": label, "ann_pct": round(ann * 100, 1),
            "sharpe": round(r.mean() / r.std() * np.sqrt(12), 2) if r.std() > 0 else None,
            "sharpe_ex_bull2425": round(exb.mean() / exb.std() * np.sqrt(12), 2) if len(exb) > 3 and exb.std() > 0 else None,
            "max_dd_pct": round(mdd * 100, 1), "yearly_pct": yearly(eq)}

# ---- stable-7 daily net on CSI300 (verbatim frozen mechanics) ----
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, volume = panel["close"], panel["volume"]
days = close.index
fwd1 = close.pct_change().shift(-1)
ret = close.pct_change()
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

from src.factors.registry import get_default_registry
reg = get_default_registry()
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
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

w = pd.DataFrame(0.0, index=days, columns=close.columns)
held = set()
for i, t in enumerate(days):
    if i % REBAL == 0:
        rowv = sig7.loc[t].dropna()
        if len(rowv) >= TOP_N:
            desired = set(rowv.nlargest(TOP_N).index)
            locked, keep = set(), held & desired
            for s in held - desired:
                if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
            buys = []
            for s in rowv.sort_values(ascending=False).index:
                if len(keep) + len(locked) + len(buys) >= TOP_N: break
                if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                buys.append(s)
            held = keep | locked | set(buys)
    if held:
        w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)
gross = (w * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
net7 = (gross - turn * 2 * COST)
n7 = net7[net7.index >= "2021-06-01"]
net7m = n7.groupby([n7.index.year, n7.index.month]).apply(lambda x: float((1 + x).prod() - 1))
net7m.index = pd.PeriodIndex([f"{y}-{m:02d}" for y, m in net7m.index], freq="M")
ml = monthly.copy()
ml.index = ml.index.to_period("M")

ov = ml.index.intersection(net7m.index)
corr_long = float(ml.loc[ov, "long_net"].corr(net7m.loc[ov]))
corr_ls = float(ml.loc[ov, "ls_net"].corr(net7m.loc[ov]))

results = [metrics(monthly["long"], "G20_long_gross"),
           metrics(monthly["long_net"], "G20_long_net"),
           metrics(monthly["ls"], "LS_gross"),
           metrics(monthly["ls_net"], "LS_net"),
           metrics(monthly["bench"], "ZZ1000_panel_EW_bench")]
out = {"note": "corrected metrics: within-year returns, ex-bull2425 Sharpe",
       "oos_months": int(len(monthly.dropna())),
       "results": results,
       "corr_vs_stable7_monthly": {"long_net": round(corr_long, 3), "ls_net": round(corr_ls, 3)},
       "stable7_same_window": metrics(net7m, "stable7_monthly_resampled")}
json.dump(out, open(DATA / "csi1000_lgbm_monthly_fixed.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

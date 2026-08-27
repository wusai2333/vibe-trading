"""Guard-fix audit on CSI300: does the corrected return guard change the
stable-7 baseline? Legacy shift(1) masked the day AFTER each anomalous
return; corrected shift(-1) masks the anomalous return itself.
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

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_and_test(mask_shift, label):
    fc = fwd.mask(anomalous.shift(mask_shift).fillna(False), 0.0)
    trad = tradable & ~anomalous
    lu = trad & (ret >= lim - 0.002)
    ld = trad & (ret <= -(lim - 0.002))
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fc.loc[t]) for t in win], index=win)) for a in STABLE7}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(rowv.nlargest(TOP_N).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not trad.at[t, s] or ld.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held: continue
                    if not trad.at[t, s] or lu.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fc.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    print(f"{label}: {cagr*100:.1f}% Sharpe {cagr/vol:.2f} MaxDD {((eq/eq.cummax())-1).min()*100:.1f}%")
    print("  yearly:", yearly)
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2),
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1), "yearly_pct": yearly}

out = {"legacy_shift1": build_and_test(1, "legacy(shift 1)"),
       "fixed_shift-1": build_and_test(-1, "fixed (shift -1)")}
json.dump(out, open(DATA / "csi300_guardfix_check.json", "w"), ensure_ascii=False, indent=1)
print("SAVED csi300_guardfix_check.json")

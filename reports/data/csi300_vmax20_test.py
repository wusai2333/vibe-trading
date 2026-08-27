"""Correlation gate + constrained incremental test for pool3_vmax20.

Gate 2 (pre-registered): max |Spearman rho| vs any stable-7 factor < 0.5
(precedent: 0.5-0.76 judged redundant in the 08-18 audit; onin20 died at
-0.77 on CSI300). Gate 3 (only if gate 2 passes): constrained-engine
incremental test, stable-7 vs stable-7+rev5, identical mechanics.
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
CORR_GATE = 0.5

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

print("computing factors...", file=sys.stderr, flush=True)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
rev5 = zscore(reg.compute("pool3_vmax20", panel).rolling(10, min_periods=6).mean())

# ---- gate 2: correlation vs stable-7 (OOS daily Spearman, mean |rho|) ----
oos = days >= OOS_START
corrs = {}
for a in STABLE7:
    rho = rev5.rank(axis=1).corrwith(fac[a].rank(axis=1), axis=1)
    corrs[a] = round(float(rho[oos].mean()), 3)
max_abs = max(abs(v) for v in corrs.values())
print("correlations vs stable-7:", json.dumps(corrs))
print(f"max |rho| = {max_abs:.3f}  gate {'PASS' if max_abs < CORR_GATE else 'FAIL'} (< {CORR_GATE})")
result = {"correlations": corrs, "max_abs_rho": round(max_abs, 3), "gate2": "pass" if max_abs < CORR_GATE else "fail"}

if max_abs >= CORR_GATE:
    json.dump(result, open(DATA / "csi300_vmax20_test.json", "w"), ensure_ascii=False, indent=1)
    sys.exit(0)

# ---- gate 3: constrained incremental test ----
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(ids, extra=None):
    allf = dict(fac)
    if extra is not None:
        allf[extra] = rev5
    idlist = ids + ([extra] if extra else [])
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    wlog = []
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([allf[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in idlist}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(allf[a].loc[blk] * wts[a] for a in idlist)
        wlog.append(wts)
    return sig, wlog

def backtest_constrained(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    ev = {"buy_blocked_limit_up": 0, "sell_locked_limit_down": 0, "sell_locked_suspended": 0}
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(rowv.nlargest(TOP_N).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s]: locked.add(s); ev["sell_locked_suspended"] += 1
                    elif limit_down.at[t, s]: locked.add(s); ev["sell_locked_limit_down"] += 1
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held: continue
                    if not tradable.at[t, s]: continue
                    elif limit_up.at[t, s]: ev["buy_blocked_limit_up"] += 1
                    else: buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    pool = fwd_clean.mean(axis=1)
    ex = (net - pool.shift(1).fillna(0))[eq.index]
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1), "yearly_pct": yearly,
            "tail_le_-4pct": int((ex <= -0.04).sum()), "worst_ex_pct": round(float(ex.min()) * 100, 2)}, net

print("gate 3: incremental constrained backtest...", file=sys.stderr, flush=True)
sig_base, _ = build_signal(STABLE7)
sig_new, wlog = build_signal(STABLE7, extra="pool3_vmax20")
base, _ = backtest_constrained(sig_base)
new, _ = backtest_constrained(sig_new)
ws = [w.get("pool3_vmax20") for w in wlog]
result["gate3"] = {"stable7": base, "stable7_plus_rev5": new,
                   "vmax20_weight": {"mean": round(float(np.mean(ws)), 3),
                                    "min": round(float(np.min(ws)), 3),
                                    "max": round(float(np.max(ws)), 3)}}
print(json.dumps(result["gate3"], ensure_ascii=False, indent=1))
json.dump(result, open(DATA / "csi300_vmax20_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_vmax20_test.json'}", file=sys.stderr)
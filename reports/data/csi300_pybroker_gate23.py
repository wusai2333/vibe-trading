"""Gate 2 + 3 for pybroker reversed factors (2026-08-21).

pvi20 (IC -0.019, t -3.3) and pvfit20 (IC -0.017, t -2.7) came out
reversed_strict in gate 1 — same shape as vol_ivol60 (in stable-7 with
negative weight). Sign-flip and run gate 2 (corr vs stable-7 < 0.5),
then gate 3 (constrained incremental blend) for whatever passes.
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
# sign-flipped reversed factors (smoothed like the rest of stable-7)
cands = {
    "pvi20_neg": -zscore(reg.compute("pybroker_pvi20", panel).rolling(10, min_periods=6).mean()),
    "pvfit20_neg": -zscore(reg.compute("pybroker_pvfit20", panel).rolling(10, min_periods=6).mean()),
}

oos = days >= OOS_START
report = {}
for name, f in cands.items():
    corrs = {a: round(float(f.rank(axis=1).corrwith(fac[a].rank(axis=1), axis=1)[oos].mean()), 3)
             for a in STABLE7}
    max_abs = max(abs(v) for v in corrs.values())
    report[name] = {"corrs": corrs, "max_abs_rho": round(max_abs, 3),
                    "gate2": "pass" if max_abs < CORR_GATE else "fail"}
    print(f"{name}: max|rho|={max_abs:.3f} -> {'PASS' if max_abs < CORR_GATE else 'FAIL'}",
          file=sys.stderr, flush=True)

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(extra_name=None, extra_f=None):
    allf = dict(fac)
    if extra_name: allf[extra_name] = extra_f
    idlist = STABLE7 + ([extra_name] if extra_name else [])
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([allf[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in idlist}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        sig.loc[days[start:start + RETRAIN]] = sum(allf[a].loc[days[start:start + RETRAIN]] * wts[a] for a in idlist)
    return sig

def backtest_constrained(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(rowv.nlargest(TOP_N).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod(); eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

print("gate 3: baseline...", file=sys.stderr, flush=True)
results = {"stable7_baseline": backtest_constrained(build_signal())}
for name, f in cands.items():
    if report[name]["gate2"] == "pass":
        print(f"gate 3: stable7+{name}...", file=sys.stderr, flush=True)
        results[f"stable7+{name}"] = backtest_constrained(build_signal(name, f))
    else:
        results[f"stable7+{name}"] = "gate2_fail"

out = {"gate2": report, "gate3": results}
json.dump(out, open(DATA / "csi300_pybroker_gate23.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

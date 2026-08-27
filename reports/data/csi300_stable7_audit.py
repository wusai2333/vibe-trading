"""stable-7 factor-quality audit: ablation + pre-registered additions.

Question: is stable-7 the right combination, or does a better one exist?

Design (anti-overfit discipline):
  * Baseline: stable-7 constrained OOS.
  * Ablation: drop each of the 7 factors one at a time — every member must
    justify its slot.
  * Addition: 5 pre-registered candidates = highest-|IR| clean-panel zoo
    factors outside stable-7, one per family branch (alpha101_054,
    alpha101_057, alpha101_025, qlib158_ksft, gtja191_111). No fishing over
    all combinations — that is how OOS gets data-mined.
  * All variants run through the constrained engine (limit/suspension +
    return guard), the only口径 that matches reality (pitfall #15).

Interpretation bar: with 7.5y daily data, CAGR differences < 2pp and Sharpe
differences < 0.05 are noise. Multiple comparisons on the same sample inflate
false discoveries — any "winner" must still prove itself on future data.
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
CANDIDATES = ["alpha101_054", "alpha101_057", "alpha101_025",
              "qlib158_ksft", "gtja191_111"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

# ---- tradability + return guard ----
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

print("computing factors...", file=sys.stderr)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
       for a in STABLE7 + CANDIDATES}

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(ids):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win))
               for a in ids}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(fac[a].loc[blk] * wts[a] for a in ids)
    return sig

def backtest_constrained(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                desired = list(rowv.nlargest(TOP_N).index); dset = set(desired)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s] or limit_down.at[t, s]:
                        locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held: continue
                    if tradable.at[t, s] and not limit_up.at[t, s]:
                        buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                  eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    return {"cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": yearly}

# ---- correlations: candidates vs stable-7 ----
def avg_corr(a, b):
    c = pd.Series([fac[a].loc[t].corr(fac[b].loc[t]) for t in days], index=days)
    return float(c.dropna().mean())
corr_tbl = {c: {s: round(avg_corr(c, s), 2) for s in STABLE7} for c in CANDIDATES}

variants = [("stable7_baseline", STABLE7)]
variants += [(f"drop_{a}", [x for x in STABLE7 if x != a]) for a in STABLE7]
variants += [(f"add_{a}", STABLE7 + [a]) for a in CANDIDATES]

results = {}
for name, ids in variants:
    print(f"running {name} ({len(ids)} factors)...", file=sys.stderr, flush=True)
    results[name] = backtest_constrained(build_signal(ids))
    r = results[name]
    print(f"  {name:28s} CAGR {r['cagr_pct']:>5.1f}% Sharpe {r['sharpe']:.2f} MaxDD {r['max_dd_pct']}%")

out = {"description": "stable-7 audit: ablation + pre-registered additions (constrained OOS)",
       "note": "CAGR diff <2pp / Sharpe diff <0.05 = noise; same-sample multiple comparisons caveat applies",
       "candidate_corr_vs_stable7": corr_tbl, "results": results}
json.dump(out, open(DATA / "csi300_stable7_audit.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_stable7_audit.json'}", file=sys.stderr)

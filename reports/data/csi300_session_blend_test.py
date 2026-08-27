"""Incremental OOS test: does session_onin20 add value to the stable-5 blend?

strict bench verdicts (clean panel): session_onin20 confirmed_alive (IR +0.113),
session_in20 reversed_strict (IR -0.104) — but corr(onin20, in20) = -0.97 (same
signal, flipped sign) and corr(onin20, qlib158_klow) = -0.77, so most of it may
already be priced in by klow's negative weight. The only honest test is a
rolling-weight OOS blend comparison with the production engine, unchanged:
z-score + rolling(10, min_periods=6), trailing-252d IR weights, 63d retrain,
Top-15 equal weight, 5d rebalance, 10bps one-way cost.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
days = close.index
fwd = close.pct_change().shift(-1)

STABLE = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060"]
CANDIDATE = "session_onin20"
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

from src.factors.registry import get_default_registry
reg = get_default_registry()

def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", file=sys.stderr)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
       for a in STABLE + [CANDIDATE]}

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(ids):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    wlog = []
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd.loc[t]) for t in win], index=win))
               for a in ids}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(fac[a].loc[blk] * wts[a] for a in ids)
        wlog.append({"applied": str(blk[0].date()), **{k: round(v, 3) for k, v in wts.items()}})
    return sig, wlog

def backtest(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    last = None
    for i, t in enumerate(sig.index):
        if last is None or i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                last = set(rowv.nlargest(TOP_N).index)
        if last:
            w.loc[t, list(last)] = 1.0 / TOP_N
    gross = (w * fwd).sum(axis=1).shift(1).fillna(0.0)
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
            "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
            "yearly_pct": yearly}

results, logs = {}, {}
for name, ids in (("stable5_baseline", STABLE),
                  ("stable5_plus_onin20", STABLE + [CANDIDATE])):
    print(f"running {name}...", file=sys.stderr)
    sig, wlog = build_signal(ids)
    results[name] = backtest(sig)
    logs[name] = wlog

# how much weight did the fit give the candidate?
cand_w = [b.get(CANDIDATE, 0.0) for b in logs["stable5_plus_onin20"]]
out = {
    "description": "incremental OOS test of session_onin20 on top of stable-5 (clean panel)",
    "candidate": {"id": CANDIDATE, "strict_category": "confirmed_alive",
                  "ir": 0.113, "corr_vs_qlib158_klow": -0.77, "corr_vs_session_in20": -0.97},
    "results": results,
    "candidate_weight_stats": {"mean": round(float(np.mean(cand_w)), 3),
                               "min": round(float(np.min(cand_w)), 3),
                               "max": round(float(np.max(cand_w)), 3)},
}
json.dump(out, open(DATA / "csi300_session_blend_test.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))
print("candidate weight mean/min/max:", out["candidate_weight_stats"])
print(f"SAVED {DATA / 'csi300_session_blend_test.json'}", file=sys.stderr)

"""Multi-factor combination test on the cached CSI300 panel.

Input: reports/data/csi300_zoo_bench.json (top factors by |IR|).
Method: rank-average (equal weight) and IC-weighted blends of the top-K
factors (deduplicated by theme), cross-sectional z-scored each day.
Backtest: long top-N, 5d rebalance, 10bps one-way cost.
Compares each blend against the best single factor and the equal-weight
benchmark.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
import numpy as np, pandas as pd

panel = pickle.load(open("reports/data/csi300_panel.pkl", "rb"))
bench = json.load(open("reports/data/csi300_zoo_bench.json"))
top_factors = [r for r in bench["top30_by_abs_ir"] if r["ir"] is not None]

from src.factors.registry import get_default_registry
reg = get_default_registry()
close = panel["close"]
fwd = close.pct_change().shift(-1)
COST, REBAL = 0.001, 5

def compute_factor(aid: str) -> pd.DataFrame:
    f = reg.compute(aid, panel)
    return f.rolling(10).mean()  # smoothing to cut ranking churn

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

def backtest(signal: pd.DataFrame, top_n: int) -> dict:
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    last = None
    for i, t in enumerate(signal.index):
        if last is None or i % REBAL == 0:
            rowv = signal.loc[t].dropna()
            if len(rowv) >= top_n:
                last = set(rowv.nlargest(top_n).index)
        if last:
            w.loc[t, list(last)] = 1.0 / top_n
    gross = (w * fwd).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    return {
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(cagr / vol, 2) if vol > 0 else None,
        "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
        "ann_turnover": round(float(turn[eq.index].sum() / yrs), 1),
    }

# --- build factor matrix: top K, dedup by zoo so families don't dominate ---
K, TOP_N = 12, 15
seen_zoo, selected = {}, []
for r in top_factors:
    if r["zoo"] in seen_zoo and seen_zoo[r["zoo"]] >= 4:
        continue
    selected.append(r)
    seen_zoo[r["zoo"]] = seen_zoo.get(r["zoo"], 0) + 1
    if len(selected) >= K:
        break
print("selected factors:", [(r["id"], r["zoo"], round(r["ir"], 3)) for r in selected], flush=True)

fac = {r["id"]: zscore(compute_factor(r["id"])) for r in selected}
print("factors computed:", len(fac), flush=True)

# --- blend 1: equal-weight rank average ---
equal = sum(f.rank(axis=1, pct=True) for f in fac.values()) / len(fac)

# --- blend 2: IC-weighted (weight = sign(ir) * |ir|) ---
weights = {r["id"]: np.sign(r["ir"]) * abs(r["ir"]) for r in selected}
wsum = sum(abs(v) for v in weights.values())
icw = sum(fac[aid] * (w / wsum) for aid, w in weights.items())

results = {}
for name, sig in (("equal_weight_blend", equal), ("ic_weighted_blend", icw)):
    results[name] = backtest(sig, TOP_N)
    print(f"{name}: {results[name]}", flush=True)

# single best factor for reference
best = selected[0]
results["best_single_" + best["id"]] = backtest(fac[best["id"]], TOP_N)
print(f"best_single_{best['id']}: {results['best_single_' + best['id']]}", flush=True)

# benchmark
bn_ret = fwd.mean(axis=1).shift(1).fillna(0.0)
bn = (1 + bn_ret).cumprod()
yrs = (close.index[-1] - close.index[0]).days / 365.25
bn_cagr = float(bn.iloc[-1] ** (1 / yrs) - 1)
results["benchmark_equal_weight"] = {
    "cagr_pct": round(bn_cagr * 100, 1),
    "sharpe": round(bn_cagr / float(fwd.mean(axis=1).std() * np.sqrt(252)), 2),
    "max_dd_pct": round(float(((bn / bn.cummax()) - 1).min()) * 100, 1),
}

out = {"selected_factors": [{k: r.get(k) for k in ("id", "zoo", "ir", "_category")} for r in selected],
       "results": results}
json.dump(out, open("reports/data/csi300_multifactor.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))

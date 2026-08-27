"""Plan A: 4-factor blend (drop reversed qlib158_klow) vs 5-factor, same OOS protocol.

Rolling 252d train / 63d retrain, weights fitted strictly on past data,
top-15 long-only, 5d rebalance, 10bps/side. Both variants on the current
panel for a fair comparison.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
import numpy as np, pandas as pd

panel = pickle.load(open("reports/data/csi300_panel.pkl", "rb"))
close = panel["close"]
fwd = close.pct_change().shift(-1)
days = close.index

PLAN_A = ["gtja191_171", "alpha101_083", "alpha101_042", "alpha101_060"]
OLD_5 = PLAN_A + ["qlib158_klow"]
TRAIN, STEP, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001

from src.factors.registry import get_default_registry
reg = get_default_registry()

def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

all_ids = sorted(set(PLAN_A) | set(OLD_5))
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in all_ids}
ic_series = {a: pd.Series([fac[a].loc[t].corr(fwd.loc[t]) for t in days[:-1]], index=days[:-1])
             for a in all_ids}
print("factors computed", flush=True)

def ir_of(ic):
    s = ic.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def rolling_blend(ids):
    blend = pd.DataFrame(np.nan, index=days, columns=close.columns)
    t = TRAIN
    while t < len(days) - 1:
        seg_end = min(t + STEP, len(days) - 1)
        irs = {a: ir_of(ic_series[a].iloc[t - TRAIN:t]) for a in ids}
        wsum = sum(abs(v) for v in irs.values())
        weights = {a: (v / wsum if wsum else 0.0) for a, v in irs.items()}
        for tt in days[t:seg_end]:
            blend.loc[tt] = sum(fac[a].loc[tt] * w for a, w in weights.items())
        t = seg_end
    return blend.loc[days[TRAIN]:].dropna(how="all")

def backtest(signal):
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    last = None
    for i, tt in enumerate(signal.index):
        if last is None or i % REBAL == 0:
            rowv = signal.loc[tt].dropna()
            if len(rowv) >= TOP_N:
                last = set(rowv.nlargest(TOP_N).index)
        if last:
            w.loc[tt, list(last)] = 1.0 / TOP_N
    gross = (w * fwd.reindex(index=signal.index)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
              for y, g in eq.groupby(eq.index.year)}
    return {"total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
            "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "ann_turnover": round(float(turn[eq.index].sum() / yrs), 1),
            "yearly_pct": yearly}

results = {}
for name, ids in (("planA_4factors", PLAN_A), ("old_5factors", OLD_5)):
    print(f"running {name}...", flush=True)
    results[name] = backtest(rolling_blend(ids))
    print(f"{name}: {results[name]}", flush=True)

json.dump({"plan_A_factors": PLAN_A, "results": results},
          open("reports/data/csi300_planA.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))

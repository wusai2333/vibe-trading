"""XGBoost blend test (2026-08-19): same pipeline as csi300_tree_blend_test.py,
swapping HistGBR for XGBRegressor(tree_method=hist). Linear baseline numbers
come from the tree test run on the SAME panel (deterministic): con 30.1%,
Sharpe 1.06, MaxDD -34.8%, IC +0.0077.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBRegressor

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
RNG_SEED = 42

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
rng = np.random.default_rng(RNG_SEED)
rfac = {f"rand{i}": pd.DataFrame(rng.standard_normal(close.shape), index=days, columns=close.columns)
        for i in range(7)}

def build_xgb(factors, depth, iters=60, lr=0.05):
    names = list(factors)
    Xall = np.stack([factors[a].values for a in names], axis=-1)
    yall = fwd_clean.values
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        tr = slice(start - TRAIN, start - 1)
        X = Xall[tr].reshape(-1, len(names))
        y = yall[tr].reshape(-1)
        ok = ~np.isnan(y)
        model = XGBRegressor(n_estimators=iters, learning_rate=lr, max_depth=depth,
                             reg_lambda=10.0, min_child_weight=300, tree_method="hist",
                             random_state=RNG_SEED, verbosity=0)
        model.fit(X[ok], y[ok])
        blk = slice(start, min(start + RETRAIN, len(days)))
        Xb = Xall[blk].reshape(-1, len(names))
        sig.iloc[blk] = model.predict(Xb).reshape(-1, close.shape[1])
    return sig

pool_ret = fwd_clean.mean(axis=1)

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
                    if s in held: continue
                    if not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

def summarize(net, sig, label):
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    ex = (net - pool_ret.shift(1).fillna(0))[eq.index]
    ic = sig.rank(axis=1).corrwith(fwd_clean.rank(axis=1), axis=1)
    ic = ic[ic.index >= OOS_START].dropna()
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": yearly,
            "tail": {"days_ex_le_-4pct": int((ex <= -0.04).sum()),
                     "worst_ex_pct": round(float(ex.min()) * 100, 2)},
            "ic": {"mean": round(float(ic.mean()), 4), "hit_pct": round(float((ic > 0).mean()) * 100, 1)}}

out = {}
for name, factors, depth in [("xgb_d2", fac, 2), ("xgb_d3", fac, 3), ("control_random_xgb", rfac, 2)]:
    print(f"running {name}...", file=sys.stderr, flush=True)
    sig = build_xgb(factors, depth)
    net = backtest_constrained(sig)
    out[name] = summarize(net, sig, name)
    r = out[name]
    print(f"  {name}: con {r['cagr_pct']}% Sharpe {r['sharpe']} MaxDD {r['max_dd_pct']}% "
          f"IC {r['ic']['mean']:+.4f} tail<=-4% {r['tail']['days_ex_le_-4pct']}")

json.dump(out, open(DATA / "csi300_xgb_blend_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_xgb_blend_test.json'}", file=sys.stderr)
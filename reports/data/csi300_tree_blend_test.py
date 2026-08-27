"""Tree-model blend test for stable-7 factors (2026-08-19, user approved).

Question: can a non-linear combiner (gradient-boosted shallow trees) extract
more from the SAME 7 factors than the production linear IR-weighted blend?

Walk-forward, lookahead-safe, mirrors production information sets exactly:
  - features: the 7 smoothed z-scored factors at t (identical to screener)
  - label: next-day guarded return; train on trailing 252d, retrain 63d
  - HistGradientBoostingRegressor (sklearn, NaN-native), no val split
  - OVERFITTING CONTROL: identical pipeline on 7 seeded Gaussian random
    features. If the random variant "finds" alpha, the pipeline is mining noise.
Backtest: constrained engine, Top-15, 5d rebalance, 10bps (same as
csi300_sector_cap_test.py baseline, which this run also reproduces).
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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

# seeded random feature stack, same shape (control)
rng = np.random.default_rng(RNG_SEED)
rfac = {f"rand{i}": pd.DataFrame(rng.standard_normal(close.shape), index=days, columns=close.columns)
        for i in range(7)}

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_linear(factors):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([factors[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in factors}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(factors[a].loc[blk] * (v / wsum) for a, v in irs.items())
    return sig

def build_tree(factors, depth, iters=60, lr=0.05):
    names = list(factors)
    Xall = np.stack([factors[a].values for a in names], axis=-1)  # (T, N, F)
    yall = fwd_clean.values
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        tr = slice(start - TRAIN, start - 1)  # days [start-252, start-2), label at t uses t+1 <= start-1
        X = Xall[tr].reshape(-1, len(names))
        y = yall[tr].reshape(-1)
        ok = ~np.isnan(y)
        model = HistGradientBoostingRegressor(max_iter=iters, learning_rate=lr, max_depth=depth,
                                              min_samples_leaf=300, l2_regularization=10.0,
                                              early_stopping=False, random_state=RNG_SEED)
        model.fit(X[ok], y[ok])  # NaN features handled natively
        blk = slice(start, min(start + RETRAIN, len(days)))
        Xb = Xall[blk].reshape(-1, len(names))
        pred = model.predict(Xb).reshape(-1, close.shape[1])
        sig.iloc[blk] = pred
    return sig

pool_ret = fwd_clean.mean(axis=1)

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
                     "worst_ex_pct": round(float(ex.min()) * 100, 2),
                     "ex_2026_le_-4pct": int((ex[ex.index.year == 2026] <= -0.04).sum())},
            "ic": {"mean": round(float(ic.mean()), 4), "hit_pct": round(float((ic > 0).mean()) * 100, 1),
                   "recent21": round(float(ic.tail(21).mean()), 4),
                   "crash_day": round(float(ic.iloc[-1]), 4)}}

out = {}
plan = [("linear", lambda: build_linear(fac)),
        ("tree_d2", lambda: build_tree(fac, 2)),
        ("tree_d3", lambda: build_tree(fac, 3)),
        ("control_random_d2", lambda: build_tree(rfac, 2))]
for name, fn in plan:
    print(f"running {name}...", file=sys.stderr, flush=True)
    sig = fn()
    net = backtest_constrained(sig)
    out[name] = summarize(net, sig, name)
    r = out[name]
    print(f"  {name}: con {r['cagr_pct']}% Sharpe {r['sharpe']} MaxDD {r['max_dd_pct']}% "
          f"IC {r['ic']['mean']:+.4f} tail<=-4% {r['tail']['days_ex_le_-4pct']}")

json.dump(out, open(DATA / "csi300_tree_blend_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_tree_blend_test.json'}", file=sys.stderr)
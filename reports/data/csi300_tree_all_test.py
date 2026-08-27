"""All-factor tree test (2026-08-19, user approved).

Hypothesis: feed the tree ALL 487 factors instead of the 7 production ones.
Pre-registered variants (no post-hoc tuning):
  tree_all_d2:    every computable registry factor, depth 2
  tree_alive_d2:  only factors the quick bench judges alive/reversed, depth 2
Same mechanics as csi300_tree_blend_test.py: 252d train / 63d retrain,
HistGBR 60 iters lr 0.05 leaf 300 L2 10, constrained Top-15/5d/10bps.
Baseline (reproduced today): linear stable-7 con 30.1% / Sharpe 1.06 / DD -34.8%.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
margin = pickle.load(open(DATA / "margin_panel.pkl", "rb"))
for k, v in margin.items():
    panel[f"margin:{k}"] = v
close = panel["close"]; volume = panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

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

# ---- alive/reversed list via quick bench (same as csi300_zoo_bench.py) ----
import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel
from src.factors.registry import get_default_registry
from src.factors.bench_runner import run_bench
reg = get_default_registry()
zoos = sorted({reg.get(aid).zoo for aid in reg.list()})
alive_ids = []
for z in zoos:
    r = run_bench(zoo=z, universe="csi300-ashare", period="2018-01-01/2026-08-19", top=20)
    for row in r.get("rows", []):
        if isinstance(row, dict) and row.get("_category") in ("alive", "reversed"):
            alive_ids.append(row["id"])
print(f"alive/reversed factors: {len(alive_ids)}", file=sys.stderr, flush=True)

# ---- compute all factors ----
print("computing all factors...", file=sys.stderr, flush=True)
fac, failed = {}, []
all_ids = sorted(reg.list())
for i, aid in enumerate(all_ids):
    try:
        fac[aid] = reg.compute(aid, panel).astype(np.float32)
    except Exception as e:
        failed.append((aid, type(e).__name__))
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_ids)} computed", file=sys.stderr, flush=True)
print(f"computed {len(fac)} factors, {len(failed)} failed: {[f[0] for f in failed][:10]}",
      file=sys.stderr, flush=True)

alive_ids = [a for a in alive_ids if a in fac]
print(f"alive subset computable: {len(alive_ids)}", file=sys.stderr, flush=True)

def build_tree(ids, depth, iters=60, lr=0.05):
    Xall = np.stack([fac[a].values for a in ids], axis=-1)
    yall = fwd_clean.values
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        tr = slice(start - TRAIN, start - 1)
        X = Xall[tr].reshape(-1, len(ids))
        y = yall[tr].reshape(-1)
        ok = ~np.isnan(y)
        Xt = X[ok]
        # drop features constant (or near-empty) within this window:
        # HistGBR binning crashes on <2 distinct values
        nn = (~np.isnan(Xt)).sum(axis=0)
        nu = np.empty(len(ids), dtype=int)
        for j in range(len(ids)):
            col = Xt[:, j]
            nu[j] = len(np.unique(col[~np.isnan(col)])) if nn[j] else 0
        keep = (nn >= 100) & (nu >= 3)
        if int(keep.sum()) < 10:
            continue
        model = HistGradientBoostingRegressor(max_iter=iters, learning_rate=lr, max_depth=depth,
                                              min_samples_leaf=300, l2_regularization=10.0,
                                              early_stopping=False, random_state=RNG_SEED)
        model.fit(Xt[:, keep], y[ok])
        blk = slice(start, min(start + RETRAIN, len(days)))
        Xb = Xall[blk].reshape(-1, len(ids))[:, keep]
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
            "tail_le_-4pct": int((ex <= -0.04).sum()),
            "ic": {"mean": round(float(ic.mean()), 4), "hit_pct": round(float((ic > 0).mean()) * 100, 1)}}

out = {"n_factors_all": len(fac), "n_factors_alive": len(alive_ids),
       "failed_factors": [f[0] for f in failed]}
for name, ids in [("tree_all_d2", sorted(fac.keys())), ("tree_alive_d2", alive_ids)]:
    t0 = time.time()
    print(f"running {name} ({len(ids)} features)...", file=sys.stderr, flush=True)
    sig = build_tree(ids, 2)
    net = backtest_constrained(sig)
    out[name] = summarize(net, sig, name)
    r = out[name]
    print(f"  {name}: con {r['cagr_pct']}% Sharpe {r['sharpe']} MaxDD {r['max_dd_pct']}% "
          f"IC {r['ic']['mean']:+.4f} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

json.dump(out, open(DATA / "csi300_tree_all_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_tree_all_test.json'}", file=sys.stderr)
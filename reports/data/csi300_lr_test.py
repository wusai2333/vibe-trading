"""Trained linear model test (2026-08-20, user: "even just an LR").

Gap closed: the production blend is IR-weighted (univariate statistics),
NOT a jointly-trained linear model. Ridge regression fits multivariate
coefficients (correlation-aware) directly on returns. Pre-registered:
  ridge7_a1:   stable-7 features, alpha=1
  ridge7_a10:  stable-7 features, alpha=10 (sensitivity)
  ridge_all_a10: all computable factors, alpha=10
Same walk-forward (252d train / 63d retrain), FIXED return guard shift(-1),
constrained Top-15/5d/10bps.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
margin = pickle.load(open(DATA / "margin_panel.pkl", "rb"))
for k, v in margin.items():
    panel["margin:" + k] = v.reindex(panel["close"].index)
event = pickle.load(open(DATA / "event_panel.pkl", "rb"))
for k, v in event.items():
    panel["event:" + k] = v.reindex(index=panel["close"].index, columns=panel["close"].columns)
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
fwd_clean = fwd.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing stable-7 factors...", file=sys.stderr, flush=True)
fac7 = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}

def build_ridge(ids, alpha, featdict):
    Xall = np.stack([featdict[a].values.astype(np.float32) for a in ids], axis=-1)
    yall = fwd_clean.values
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        tr = slice(start - TRAIN, start - 1)
        X = Xall[tr].reshape(-1, len(ids))
        y = yall[tr].reshape(-1)
        ok = ~np.isnan(y)
        m = Ridge(alpha=alpha)
        m.fit(np.nan_to_num(X[ok], nan=0.0), y[ok])
        blk = slice(start, min(start + RETRAIN, len(days)))
        Xb = Xall[blk].reshape(-1, len(ids))
        sig.iloc[blk] = m.predict(np.nan_to_num(Xb, nan=0.0)).reshape(-1, close.shape[1])
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

def summarize(net, label):
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    ex = (net - pool_ret.shift(1).fillna(0))[eq.index]
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": yearly,
            "tail_le_-4pct": int((ex <= -0.04).sum()),
            "ex_mean_pct": round(float(ex.mean()) * 100, 3)}

out = {}
for name, ids, alpha in [("ridge7_a1", STABLE7, 1.0), ("ridge7_a10", STABLE7, 10.0)]:
    t0 = time.time()
    print("running " + name, file=sys.stderr, flush=True)
    sig = build_ridge(ids, alpha, fac7)
    out[name] = summarize(backtest_constrained(sig), name)
    r = out[name]
    print("  " + name + ": " + str(r['cagr_pct']) + "% Sh " + str(r['sharpe'])
          + " DD " + str(r['max_dd_pct']) + "% (" + str(round(time.time()-t0)) + "s)", file=sys.stderr, flush=True)

# all-factor variant: compute every registered factor
print("computing all factors...", file=sys.stderr, flush=True)
fac = dict(fac7)
failed = []
for aid in sorted(reg.list()):
    if aid in fac: continue
    try:
        fac[aid] = zscore(reg.compute(aid, panel).rolling(10, min_periods=6).mean())
    except Exception as e:
        failed.append(aid)
all_ids = sorted(fac.keys())
print("factors: " + str(len(all_ids)) + " (failed " + str(len(failed)) + ")", file=sys.stderr, flush=True)
t0 = time.time()
sig = build_ridge(all_ids, 10.0, fac)
out["ridge_all_a10"] = summarize(backtest_constrained(sig), "ridge_all_a10")
r = out["ridge_all_a10"]
print("  ridge_all_a10: " + str(r['cagr_pct']) + "% Sh " + str(r['sharpe'])
      + " DD " + str(r['max_dd_pct']) + "% (" + str(round(time.time()-t0)) + "s)", file=sys.stderr, flush=True)

json.dump(out, open(DATA / "csi300_lr_test.json", "w"), ensure_ascii=False, indent=1)
print("SAVED csi300_lr_test.json", file=sys.stderr)

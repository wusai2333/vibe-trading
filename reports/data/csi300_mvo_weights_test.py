"""MVO weight layer on frozen stable-7 (astock-alpha-factor-lab borrowed spec).

Last untested pipeline layer: selection (13 lines) and timing (regime) are
closed; weighting was always equal-weight. Borrow portfolio_opt.py's recipe
verbatim: Ledoit-Wolf shrinkage covariance (252d trailing, PIT), then

  EW     equal weight (baseline, should reproduce 29.4/1.04/-34.5)
  InvVol inverse-volatility
  MinVar minimum variance (they warn it drops alpha — included for record)
  MVO    maximize standardized signal alpha s.t. variance <= EW variance,
         long-only, fully invested, per-name cap 0.08 (their CAP verbatim),
         CLARABEL->SCS fallback, EW fallback on solver failure

Applied at each 5d rebalance to the SAME held books the frozen constrained
simulator picks (limit/suspension mechanics untouched); weights static within
a block. Costs on all weight changes. No lookahead: covariance uses returns
through close[t], anomalous cells zeroed.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, volume = panel["close"], panel["volume"]
days = close.index
fwd1 = close.pct_change().shift(-1)
ret = close.pct_change()

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")
CAP, LOOKBACK = 0.08, 252

tradable = close.notna() & volume.fillna(0).gt(0)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd1_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))
ret_clean = ret.mask(anomalous, 0.0)

from src.factors.registry import get_default_registry
reg = get_default_registry()

def cz(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", file=sys.stderr)
fac7 = {a: cz(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac7[a].loc[t].corr(fwd1_clean.loc[t]) for t in win], index=win))
           for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    sig7.loc[days[start:start + RETRAIN]] = sum(fac7[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)

# ---- constrained simulator: record held set + signal per rebalance ----
rebalances = []  # (t_idx, held_list)
held = set()
for i, t in enumerate(days):
    if i % REBAL == 0:
        rowv = sig7.loc[t].dropna()
        if len(rowv) >= TOP_N:
            desired = set(rowv.nlargest(TOP_N).index)
            locked, keep = set(), held & desired
            for s in held - desired:
                if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
            buys = []
            for s in rowv.sort_values(ascending=False).index:
                if len(keep) + len(locked) + len(buys) >= TOP_N: break
                if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                buys.append(s)
            held = keep | locked | set(buys)
            rebalances.append((i, list(held)))
print(f"{len(rebalances)} rebalances recorded", file=sys.stderr)

def solve(kind, Sigma, alpha, var_budget):
    n = len(alpha)
    w = cp.Variable(n)
    S = cp.psd_wrap(Sigma)
    cons = [w >= 0, cp.sum(w) == 1, w <= CAP]
    if kind == "minvar":
        obj = cp.Minimize(cp.quad_form(w, S))
    else:  # mvo
        cons.append(cp.quad_form(w, S) <= var_budget)
        obj = cp.Maximize(alpha @ w)
    prob = cp.Problem(obj, cons)
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and np.isfinite(w.value).all():
                wv = np.clip(w.value, 0, None)
                return wv / wv.sum() if wv.sum() > 0 else None
        except Exception:
            continue
    return None

def build_weights(mode: str) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=days, columns=close.columns)
    stats = {"solved": 0, "fallback_ew": 0}
    for bi, (i, held_l) in enumerate(rebalances):
        t = days[i]
        end = days[rebalances[bi + 1][0] - 1] if bi + 1 < len(rebalances) else days[-1]
        n = len(held_l)
        w_ew = pd.Series(1.0 / max(n, TOP_N), index=held_l)
        target = w_ew
        if mode != "EW" and n >= 5:
            win = ret_clean.loc[:t].iloc[-LOOKBACK:][held_l]
            if win.notna().all().all() and len(win) >= 200:
                Sigma = LedoitWolf().fit(win.values).covariance_
                a = sig7.loc[t, held_l].values.astype(float)
                a = (a - a.mean()) / (a.std() + 1e-9)
                var_ew = float(w_ew.values @ Sigma @ w_ew.values)
                if mode == "InvVol":
                    vol = np.sqrt(np.diag(Sigma))
                    wv = 1.0 / vol
                    wv = np.minimum(wv / wv.sum(), CAP)
                    target = pd.Series(wv / wv.sum(), index=held_l)
                else:
                    wv = solve("minvar" if mode == "MinVar" else "mvo", Sigma, a, var_ew)
                    if wv is not None:
                        target = pd.Series(wv, index=held_l)
                        stats["solved"] += 1
                    else:
                        stats["fallback_ew"] += 1
            else:
                stats["fallback_ew"] += 1
        w.loc[t:end, held_l] = target.values
    return w, stats

def stats_from(net: pd.Series, label: str) -> dict:
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    mdd = float(((eq / eq.cummax()) - 1).min())
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(mdd * 100, 1), "calmar": round(cagr / abs(mdd), 2) if mdd < 0 else None}

results, all_stats = [], {}
for mode in ["EW", "InvVol", "MinVar", "MVO"]:
    print(f"building {mode}...", file=sys.stderr)
    w, opt_stats = build_weights(mode)
    gross = (w * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    results.append(stats_from(net, mode))
    all_stats[mode] = opt_stats

out = {"description": "MVO weight layer (astock-lab spec verbatim) on frozen stable-7 books",
       "spec": {"cov": "Ledoit-Wolf 252d", "cap": CAP, "mvo": "max alpha s.t. var<=EW var",
                "fallback": "EW"},
       "results": results, "optimizer_stats": all_stats}
json.dump(out, open(DATA / "csi300_mvo_weights.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

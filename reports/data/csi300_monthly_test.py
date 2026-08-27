"""Monthly-rebalance track for long-horizon factors (2026-08-20).

Pre-registered: the 4 monthly-IC survivors (dnbeta120 +0.034, carhart_mom
+0.032, fund_earnings_yield +0.024, raw_mom252 +0.021). Mechanics: monthly
(20d) rebalance, Top-30 equal weight, constrained engine, 10bps one-way.
Variants: monthly_blend (rolling-252d monthly-horizon IR weights, 63d
retrain) and monthly_equal (equal weight, robustness). Pass bar: OOS
Sharpe >= 1.0 with positive pool excess.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for k, v in fund.items():
    if k.startswith("fund:"):
        panel[k] = v
close = panel["close"]; volume = panel["volume"]
days = close.index
ret = close.pct_change()
fwd1 = ret.shift(-1)
fwd20 = close.pct_change(20).shift(-20)

FACTORS = ["lit_dnbeta120", "academic_carhart_mom", "fund_earnings_yield", "raw_mom252"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 30, 20, 0.001
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
fwd_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()
print("computing factors...", file=sys.stderr, flush=True)
fac = {}
for a in ["lit_dnbeta120", "academic_carhart_mom", "fund_earnings_yield"]:
    fac[a] = reg.compute(a, panel)
fac["raw_mom252"] = close.pct_change(252)
# cross-sectional z-score per day
for a in FACTORS:
    mu, sd = fac[a].mean(axis=1), fac[a].std(axis=1)
    fac[a] = fac[a].sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

print("building signals...", file=sys.stderr, flush=True)
sig_blend = pd.DataFrame(np.nan, index=days, columns=close.columns)
sig_equal = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN + 20, len(days), RETRAIN):
    # PIT: fwd20 at window day t covers t..t+20, so the window must end
    # 20 days before the block starts (no overlap with predicted days).
    win = days[start - TRAIN - 20:start - 20]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd20.loc[t]) for t in win], index=win)) for a in FACTORS}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + RETRAIN]
    sig_blend.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())
    sig_equal.loc[blk] = sum(fac[a].loc[blk] for a in FACTORS) / len(FACTORS)

pool_ret = fwd_clean.mean(axis=1)

def backtest_monthly(sig):
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
                    if not tradable.at[t, s] or limit_up.at[t, s]:
                        if limit_up.at[t, s]: ev["buy_blocked_limit_up"] += 1
                        continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST, ev

def summarize(net, label):
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    ex = (net - pool_ret.shift(1).fillna(0))[eq.index]
    ex_ann = float(ex.mean()) * 252
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": yearly,
            "ex_ann_pct": round(ex_ann * 100, 1)}

out = {}
pe = pool_ret[pool_ret.index >= OOS_START]
out["pool_ew_cagr_pct"] = round(float((1 + pe).prod() ** (1 / ((pe.index[-1] - OOS_START).days / 365.25)) - 1) * 100, 1)
for name, sig in [("monthly_blend", sig_blend), ("monthly_equal", sig_equal)]:
    t0 = time.time()
    net, ev = backtest_monthly(sig)
    out[name] = summarize(net, name)
    out[name + "_events"] = ev
    r = out[name]
    print("  " + name + ": " + str(r['cagr_pct']) + "% Sh " + str(r['sharpe'])
          + " DD " + str(r['max_dd_pct']) + "% ex " + str(r['ex_ann_pct']) + "pp/yr ("
          + str(round(time.time() - t0)) + "s)", file=sys.stderr, flush=True)
json.dump(out, open(DATA / "csi300_monthly_test.json", "w"), ensure_ascii=False, indent=1)
print("SAVED csi300_monthly_test.json", file=sys.stderr)

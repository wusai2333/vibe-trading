"""Incremental OOS test of limit/vol candidates on top of stable-5.

Candidates (strict verdicts on the clean panel):
  limit_dist   confirmed_alive  IR +0.155 (IC ~100x random), max corr vs stable-5 0.20
  vol_ivol60   reversed_strict  IR -0.092 (IC ~48x random), max corr vs stable-5 0.40
  (limit_upcnt20/limit_dncnt20/vol_rvol20 rejected: weak or redundant, rvol~ivol corr 0.83)

Runs each variant through BOTH engines, because factors that tilt toward
limit-up stocks earn phantom returns in the unconstrained engine (you cannot
buy a sealed stock):
  * unconstrained: original top-15 mechanics
  * constrained:   limit/suspension position simulator + return guard
Same signal in both: z-score + rolling(10, min_periods=6), trailing-252d IR
weights (vs guarded fwd), 63d retrain, Top-15, 5d rebalance, 10bps one-way.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi500_panel.pkl", "rb"))
_fund = pickle.load(open(DATA / "fund_cache_csi500.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"): panel[_k] = _v
close = panel["close"]; volume = panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

STABLE = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

# ---- tradability + return guard (same as csi300_constrained_backtest.py) ----
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

ALL = STABLE + ["fund_earnings_yield"]
print("computing factors...", file=sys.stderr)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in ALL}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(ids):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    wlog = []
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in ids}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        blk = days[start:start + RETRAIN]
        sig.loc[blk] = sum(fac[a].loc[blk] * wts[a] for a in ids)
        wlog.append(wts)
    return sig, wlog

def stats_from(net, w, label):
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    return {"label": label, "cagr_pct": round(cagr*100,1),
            "sharpe": round(cagr/vol,2) if vol>0 else None,
            "max_dd_pct": round(float(((eq/eq.cummax())-1).min())*100,1),
            "yearly_pct": yearly}

def backtest_unconstrained(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    last = None
    for i, t in enumerate(sig.index):
        if last is None or i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                last = set(rowv.nlargest(TOP_N).index)
        if last:
            w.loc[t, list(last)] = 1.0 / TOP_N
    gross = (w * fwd_clean).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return stats_from(gross - turn * 2 * COST, w, "unconstrained")

def backtest_constrained(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    ev = {"buy_blocked_limit_up": 0, "buy_blocked_suspended": 0,
          "sell_locked_limit_down": 0, "sell_locked_suspended": 0}
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                desired = list(rowv.nlargest(TOP_N).index); dset = set(desired)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s]: locked.add(s); ev["sell_locked_suspended"] += 1
                    elif limit_down.at[t, s]: locked.add(s); ev["sell_locked_limit_down"] += 1
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held: continue
                    if not tradable.at[t, s]: ev["buy_blocked_suspended"] += 1
                    elif limit_up.at[t, s]: ev["buy_blocked_limit_up"] += 1
                    else: buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return stats_from(gross - turn * 2 * COST, w, "constrained"), ev

VARIANTS = {"stable7": STABLE,
            "plus_ep": STABLE + ["fund_earnings_yield"]}
out = {}
for name, ids in VARIANTS.items():
    print(f"running {name}...", file=sys.stderr, flush=True)
    sig, wlog = build_signal(ids)
    unc = backtest_unconstrained(sig)
    con, ev = backtest_constrained(sig)
    wstats = {}
    for extra in ("fund_earnings_yield",):
        ws = [b.get(extra) for b in wlog if extra in b]
        if ws:
            wstats[extra] = {"mean": round(float(np.mean(ws)), 3),
                             "min": round(float(np.min(ws)), 3),
                             "max": round(float(np.max(ws)), 3)}
    out[name] = {"unconstrained": unc, "constrained": con,
                 "constraint_events": ev, "candidate_weights": wstats}
    print(f"  {name}: unc {unc['cagr_pct']}% / con {con['cagr_pct']}% "
          f"(Sharpe {con['sharpe']}, 涨停挡买 {ev['buy_blocked_limit_up']} 次)")

json.dump(out, open(DATA / "csi500_ep_test.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: {"unc": v["unconstrained"]["cagr_pct"], "con": v["constrained"]["cagr_pct"],
                      "con_sharpe": v["constrained"]["sharpe"]} for k, v in out.items()},
      ensure_ascii=False, indent=1))
print(f"SAVED {DATA / 'csi500_ep_test.json'}", file=sys.stderr)

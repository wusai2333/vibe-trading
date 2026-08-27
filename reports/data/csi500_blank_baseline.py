"""CSI500 blank-baseline pilot: build a native combo from validated candidates.

Question: does a CSI500-native candidate pool beat the PORTED stable-7
(constrained 42.2% / Sharpe 1.32, measured in csi500_ep_test.json) under
identical mechanics (rolling-252d IR weights, 63d retrain, min_periods=6,
Top-15/5d rebalance/10bps, dual engines)?

Candidate pool = everything that survived strict in the two pilot rounds:
  alive:    session_onin20 (0.200), limit_dist (0.154), session_on20 (0.114),
            session_on5 (0.110), fund_earnings_yield (0.077)
  reversed: session_in20 (-0.180), vol_ivol60 (-0.149), vol_rvol20 (-0.125),
            limit_upcnt20 (-0.125), limit_dncnt20 (-0.104)
Reversed factors enter as-is; the rolling IR fit assigns negative weights
(same mechanism that makes ivol60 work in stable-7).

Pre-specified variants (no data-driven selection, to avoid selection overfit):
  pool_top3   : onin20 + limit_dist + ivol60          (the recommendation)
  pool_alive  : the 5 alive factors
  pool_all    : all 10 candidates, let IR weighting pick

Caveats: survivorship-biased universe (current constituents); comparison is
"native pool vs port under identical conditions", not a production claim.
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
    if _k.startswith("fund:"):
        panel[_k] = _v
close = panel["close"]; volume = panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

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

ALIVE = ["session_onin20", "limit_dist", "session_on20", "session_on5",
         "fund_earnings_yield"]
REVERSED = ["session_in20", "vol_ivol60", "vol_rvol20", "limit_upcnt20",
            "limit_dncnt20"]
ALL = ALIVE + REVERSED

_cache_path = DATA / "csi500_factor_cache.pkl"
_cache = pickle.load(open(_cache_path, "rb"))
for a in ALL:
    if a not in _cache:
        _cache[a] = reg.compute(a, panel)
pickle.dump(_cache, open(_cache_path, "wb"))
print(f"factor cache: {len(_cache)} factors", file=sys.stderr)

fac = {a: zscore(_cache[a].rolling(10, min_periods=6).mean()) for a in ALL}

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
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
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

VARIANTS = {
    "pool_top3": ["session_onin20", "limit_dist", "vol_ivol60"],
    "pool_alive": ALIVE,
    "pool_all": ALL,
}
out = {}
for name, ids in VARIANTS.items():
    print(f"running {name} ({len(ids)} factors)...", file=sys.stderr, flush=True)
    sig, wlog = build_signal(ids)
    unc = backtest_unconstrained(sig)
    con, ev = backtest_constrained(sig)
    wstats = {}
    for a in ids:
        ws = [b.get(a) for b in wlog if b.get(a) is not None]
        if ws:
            wstats[a] = {"mean": round(float(np.mean(ws)), 3),
                         "min": round(float(np.min(ws)), 3),
                         "max": round(float(np.max(ws)), 3)}
    out[name] = {"unconstrained": unc, "constrained": con,
                 "constraint_events": ev, "weights": wstats}
    print(f"  {name}: unc {unc['cagr_pct']}% / con {con['cagr_pct']}% "
          f"(Sharpe {con['sharpe']}, MaxDD {con['max_dd_pct']}%, 涨停挡买 {ev['buy_blocked_limit_up']} 次)")

json.dump(out, open(DATA / "csi500_blank_baseline.json", "w"),
          ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi500_blank_baseline.json'}", file=sys.stderr)

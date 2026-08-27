"""CSI1000 port test: stable-7 dropped unchanged onto the 1000-stock pool.

Round 1 of the pool-expansion pilot (user question: is 300 too small?).
Pre-registered: cost 15bps one-way primary (small-cap reality), 10bps as
sensitivity; benchmark = CSI1000 equal-weight pool; pass bar = OOS Sharpe
>= 1.0 with clear pool-beating excess. Same mechanics as production:
rolling-252d IR weights, 63d retrain, min_periods=6 smoothing, Top-15,
5d rebalance, constrained engine (limit/suspension gates + return guard).
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi1000_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)
ret = close.pct_change()
print(f"panel: {close.shape[1]} names x {len(days)} days", file=sys.stderr)

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL = 252, 63, 15, 5
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
# fwd[t] carries ret[t+1]: mask where THAT return is anomalous (shift(-1)).
# Legacy shift(1) masked the day AFTER the bad return and let the bad return
# itself into P&L (found via CSI1000 2023 +243% artifact).
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
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

print("building signal...", file=sys.stderr, flush=True)
sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + RETRAIN]
    sig.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())

pool_ret = fwd_clean.mean(axis=1)

def backtest_constrained(cost):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    ev = {"buy_blocked_limit_up": 0, "buy_blocked_suspended": 0,
          "sell_locked_limit_down": 0, "sell_locked_suspended": 0}
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
                    if not tradable.at[t, s]: ev["buy_blocked_suspended"] += 1
                    elif limit_up.at[t, s]: ev["buy_blocked_limit_up"] += 1
                    else: buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * cost, ev

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
            "worst_ex_pct": round(float(ex.min()) * 100, 2),
            "ex_mean_pct": round(float(ex.mean()) * 100, 3)}

out = {"pool_ew_cagr_pct": round(float(((1 + pool_ret[pool_ret.index >= OOS_START]).cumprod().iloc[-1])
                                       ** (1 / ((pool_ret.index[-1] - OOS_START).days / 365.25)) - 1) * 100, 1)}
for bps in (15, 10):
    t0 = time.time()
    net, ev = backtest_constrained(bps / 10000)
    out[f"constrained_{bps}bps"] = summarize(net, f"con_{bps}bps")
    out[f"events_{bps}bps"] = ev
    r = out[f"constrained_{bps}bps"]
    print(f"  {bps}bps: {r['cagr_pct']}% Sharpe {r['sharpe']} MaxDD {r['max_dd_pct']}% "
          f"ex {r['ex_mean_pct']}%/d ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

json.dump(out, open(DATA / "csi1000_port_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi1000_port_test.json'}", file=sys.stderr)
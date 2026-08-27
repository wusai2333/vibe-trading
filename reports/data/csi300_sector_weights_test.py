"""Per-sector factor-weight test (2026-08-19, user hypothesis).

Hypothesis: instead of ONE global IR-weighted blend, fit factor weights
PER SECTOR (each stock scored by the rolling IR weights of its own sector).
Pre-registered variants:
  baseline:      global stable-7 mechanics (reproduction)
  sector_w:      per-sector IR weights, global cross-sector ranking
  sector_shrink: 50/50 shrinkage of sector weights toward global weights
                 (pre-registered hedge for tiny sectors: energy 8, utilities 11)
Same factors (stable-7 seven), same retrain cadence, constrained engine.
Caveat: sector membership = current csindex cache applied historically.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

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

sec_map = json.load(open(DATA / "stock2sector_cache.json"))
sym_sec = pd.Series({s: sec_map.get(s.split(".")[0], "其他") for s in close.columns})
sectors = sorted(set(sym_sec))
sec_members = {s: list(sym_sec[sym_sec == s].index) for s in sectors}
print("sectors:", {s: len(v) for s, v in sec_members.items()}, file=sys.stderr)

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

def fit_weights(win, members=None):
    irs = {}
    for a in STABLE7:
        if members is None:
            ic = pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)
        else:
            ic = pd.Series([fac[a].loc[t, members].corr(fwd_clean.loc[t, members]) for t in win], index=win)
        irs[a] = ir_of(ic)
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    return {a: v / wsum for a, v in irs.items()}

def build_signal(mode):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        gw = fit_weights(win)
        if mode == "baseline":
            sw = {s: gw for s in sectors}
        else:
            sw = {}
            for s in sectors:
                w = fit_weights(win, sec_members[s])
                if mode == "sector_shrink":
                    w = {a: 0.5 * w[a] + 0.5 * gw[a] for a in STABLE7}
                sw[s] = w
        blk = days[start:start + RETRAIN]
        for s in sectors:
            m = sec_members[s]
            sig.loc[blk, m] = sum(fac[a].loc[blk, m] * sw[s][a] for a in STABLE7)
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
            "worst_ex_pct": round(float(ex.min()) * 100, 2)}

out = {}
for mode in ("baseline", "sector_w", "sector_shrink"):
    t0 = time.time()
    print(f"running {mode}...", file=sys.stderr, flush=True)
    sig = build_signal(mode)
    net = backtest_constrained(sig)
    out[mode] = summarize(net, mode)
    r = out[mode]
    print(f"  {mode}: con {r['cagr_pct']}% Sharpe {r['sharpe']} MaxDD {r['max_dd_pct']}% "
          f"tail {r['tail_le_-4pct']} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

json.dump(out, open(DATA / "csi300_sector_weights_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_sector_weights_test.json'}", file=sys.stderr)
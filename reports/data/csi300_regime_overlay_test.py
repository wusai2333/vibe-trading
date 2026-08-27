"""Regime overlay on frozen stable-7 (FinRL-Trading borrowed thresholds).

Question: can a portfolio-level risk overlay cut stable-7's -34.8% MaxDD
without touching the frozen model? Borrow FinRL-X's adaptive_rotation regime
spec VERBATIM (no tuning on our data), map SPX/VIX -> CSI300 index:

  slow regime (daily, FinRL weekly x5):
    trend : idx < 130d MA            (26-week MA)
    dd    : 65d drawdown <= -10%     (13-week, 10%)
    volz  : robust-z(20d realized vol, 756d median/MAD window) >= 3.0
    score = sum(0..3) -> exposure 1.0 / 0.7 / 0.5  (score 0 / 1 / >=2)
    persistence: raw state must hold 10 days before switching (2 weeks)
  fast risk-off:
    3d index return <= -3%  OR  (volz >= 3 AND robust-z(daily dvolz) >= 3.5)
    -> exposure 0.3 for 10 days, overrides slow

Timing: m[t] uses index data through close[t]; w[t] set at close[t];
P&L earned on fwd[t] (t -> t+1). No lookahead. Costs on every scaling trade.

Variants (pre-registered, not tuned):
  V0 baseline constrained stable-7
  V1 slow only
  V2 slow + fast
  V3 binary slow (score==0 -> 1.0 else 0.5)
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, volume = panel["close"], panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

# ---- tradability + guard (verbatim from frozen pipeline) ----
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

print("computing factors...", file=sys.stderr)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

signal = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win))
           for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    signal.loc[days[start:start + RETRAIN]] = sum(fac[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)

# ---- constrained position simulator (verbatim) ----
w = pd.DataFrame(0.0, index=days, columns=close.columns)
held = set()
for i, t in enumerate(days):
    if i % REBAL == 0:
        rowv = signal.loc[t].dropna()
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
    if held:
        w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)

def net_of(wm: pd.DataFrame) -> pd.Series:
    gross = (wm * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (wm.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

# ---- regime signals from CSI300 index ----
idx = pd.read_csv(DATA / "csi300_index_daily.csv", parse_dates=["date"]).set_index("date")["close"]
idx = idx.reindex(days).ffill()
ma130 = idx.rolling(130).mean()
dd65 = idx / idx.rolling(65).max() - 1
rvol = np.log(idx / idx.shift(1)).rolling(20).std() * np.sqrt(252)

def robust_z(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window, min_periods=252).median()
    mad = (x - med).abs().rolling(window, min_periods=252).median()
    return (x - med) / (1.4826 * mad.replace(0, np.nan))

volz = robust_z(rvol, 756)
dvolz_z = robust_z(volz.diff(), 756)

trend = idx < ma130
ddst = dd65 <= -0.10
vst = volz >= 3.0
score = (trend.astype(int) + ddst.astype(int) + vst.astype(int)).fillna(0)

def persist(raw: pd.Series, n: int = 10) -> pd.Series:
    """effective state: switch only after raw held n consecutive days"""
    out, cur, cnt = [], np.nan, 0
    prev_raw = np.nan
    for v in raw:
        if v == prev_raw: cnt += 1
        else: cnt = 1
        prev_raw = v
        if pd.isna(cur) or v == cur: cur = v
        elif cnt >= n: cur = v
        out.append(cur)
    return pd.Series(out, index=raw.index)

SLOW_MAP = {0: 1.0, 1: 0.7}
m_slow3 = persist(score).map(lambda s: SLOW_MAP.get(int(s), 0.5))
m_bin = persist((score >= 2).astype(int)).map({0: 1.0, 1: 0.5})

shock_price = idx.pct_change(3) <= -0.03
shock_vol = (volz >= 3.0) & (dvolz_z >= 3.5)
fast = shock_price | shock_vol
fast_active = pd.Series(0, index=days)
until = -1
for i, t in enumerate(days):
    if fast.iloc[i]: until = i + 10
    fast_active.iloc[i] = 1 if i < until else 0
m_slowfast = m_slow3.where(fast_active == 0, 0.3)

def stats(net: pd.Series, label: str) -> dict:
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    mdd = float(((eq / eq.cummax()) - 1).min())
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(mdd * 100, 1),
            "calmar": round(cagr / abs(mdd), 2) if mdd < 0 else None,
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                                eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

base_net = net_of(w)
variants = {
    "V0_baseline": base_net,
    "V1_slow": net_of(w.multiply(m_slow3, axis=0)),
    "V2_slow_fast": net_of(w.multiply(m_slowfast, axis=0)),
    "V3_binary": net_of(w.multiply(m_bin, axis=0)),
}
results = [stats(v, k) for k, v in variants.items()]

oos = days[days >= OOS_START]
occ = {s: round(float((persist(score).reindex(oos) == s).mean()) * 100, 1)
       for s in [0, 1, 2, 3]}
diag = {"regime_occupancy_pct_oos": {"score0_risk_on": occ[0], "score1_neutral": occ[1],
                                     "score2": occ[2], "score3": occ[3]},
        "fast_days_oos": int(fast_active.reindex(oos).sum()),
        "avg_exposure_oos": {k: round(float(m.reindex(oos).mean()), 3)
                             for k, m in [("V1", m_slow3), ("V2", m_slowfast), ("V3", m_bin)]},
        "exposure_switches_oos": {k: int((m.reindex(oos).diff().fillna(0) != 0).sum())
                                  for k, m in [("V1", m_slow3), ("V2", m_slowfast), ("V3", m_bin)]}}

out = {"description": "FinRL-X regime overlay (verbatim thresholds) on frozen stable-7",
       "thresholds": {"trend": "idx<MA130", "dd": "65d<=-10%", "volz": ">=3 (756d robust)",
                      "persistence_days": 10, "fast": "3d<=-3% or volz>=3&dvolz_z>=3.5, 10d",
                      "exposures": "1.0/0.7/0.5, fast 0.3"},
       "results": results, "diagnostics": diag}
json.dump(out, open(DATA / "csi300_regime_overlay.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))
print(json.dumps(diag, ensure_ascii=False, indent=1))

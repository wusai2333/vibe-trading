"""CST-quant claims on our bench: practice is the sole criterion (2026-08-21).

Cheap-Stable-Trending-quant repo makes five empirical claims. Test each on
OUR data (CSI300 clean panel + fund_cache) under OUR gates:

  C1 momentum window: 61-21 (40d) beats 12-1m and 6-1m on A-shares (their P5)
  C2 EP is the strongest value factor (their P1/P2, FM t=3.41 quarterly)
  C3 LowVol is a RISK-ADJUSTER: weak standalone IC but low-corr portfolio
     value (their F5 lesson — directly challenges our gate-1 IC bar)
  C4 ROE is negative/zero alpha on A-shares (their P1-F4)
  C5 their production trio (EP+LowVol40+MOM40 equal-z) as a sleeve:
     both our mechanism (daily top-15/5d) and theirs (quarterly top-50 EW)

Factors: EP/ROE via existing zoo (fund_earnings_yield / fund_roe — PIT by
pubDate); price factors from panel. Gate-2/3 protocol verbatim from
csi300_vmax20_test.py.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(days).reindex(columns=close.columns)
fwd = close.pct_change().shift(-1)
ret = close.pct_change()

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")
CORR_GATE = 0.5

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
fac7 = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}

cst = {
    "ep": zscore(reg.compute("fund_earnings_yield", panel)),
    "roe": zscore(reg.compute("fund_roe", panel)),
    "lowvol40": zscore(-ret.rolling(40).std()),
    "mom40": zscore(close.shift(21) / close.shift(61) - 1),
    "mom_12_1": zscore(close.shift(21) / close.shift(273) - 1),
    "mom_6_1": zscore(close.shift(21) / close.shift(147) - 1),
}
trio = zscore((cst["ep"] + cst["lowvol40"] + cst["mom40"]) / 3)

# ---- A: daily IC (our yardstick) ----
oos_days = days[days >= OOS_START]
def daily_ic(f):
    ic = pd.Series({t: f.loc[t].corr(fwd_clean.loc[t], method="spearman") for t in oos_days}).dropna()
    return {"ic_mean": round(float(ic.mean()), 4),
            "ic_ir": round(float(ic.mean() / ic.std()), 3) if ic.std() else None}
print("daily IC...", file=sys.stderr, flush=True)
claim_ic = {k: daily_ic(v) for k, v in cst.items()}
claim_ic["trio_composite"] = daily_ic(trio)

# ---- B: gate-2 correlation vs stable-7 ----
corrs = {}
for k in ["ep", "lowvol40", "mom40"]:
    rhos = {a: float(cst[k].rank(axis=1).corrwith(fac7[a].rank(axis=1), axis=1)[days >= OOS_START].mean())
            for a in STABLE7}
    corrs[k] = {"vs_each": {a: round(v, 3) for a, v in rhos.items()},
                "max_abs": round(max(abs(v) for v in rhos.values()), 3)}

# ---- C: gate-3 incremental blend (vmax20 protocol) ----
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(ids, extra_name=None, extra_f=None):
    allf = dict(fac7)
    if extra_name: allf[extra_name] = extra_f
    idlist = ids + ([extra_name] if extra_name else [])
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([allf[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in idlist}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        wts = {a: v / wsum for a, v in irs.items()}
        sig.loc[days[start:start + RETRAIN]] = sum(allf[a].loc[days[start:start + RETRAIN]] * wts[a] for a in idlist)
    return sig

def net_constrained(sig, top_n=TOP_N, rebal=REBAL):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % rebal == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= top_n:
                dset = set(rowv.nlargest(top_n).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= top_n: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), top_n)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

def stats_constrained(net):
    eq = (1 + net).cumprod(); eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

def backtest_constrained(sig, top_n=TOP_N, rebal=REBAL):
    return stats_constrained(net_constrained(sig, top_n, rebal))

print("gate 3 blends...", file=sys.stderr, flush=True)
sig_base = build_signal(STABLE7)
blends = {}
for k in ["ep", "lowvol40", "mom40"]:
    if corrs[k]["max_abs"] < CORR_GATE:
        blends[f"stable7+{k}"] = backtest_constrained(build_signal(STABLE7, k, cst[k]))
    else:
        blends[f"stable7+{k}"] = "gate2_fail"
blends["stable7_baseline"] = backtest_constrained(sig_base)

# ---- D: trio standalone, both mechanisms ----
print("trio sleeves...", file=sys.stderr, flush=True)
sleeves = {"trio_daily_top15_5d": backtest_constrained(trio),
           "trio_quarterly_top50": backtest_constrained(trio, top_n=50, rebal=63)}

# ---- E: risk-adjuster channels (putting OUR gates on trial) ----
# CST's F5 lesson: weak-IC low-corr factors may add value through portfolio
# construction, not signal blending. Gate-3 cannot see that channel. Test it:
#   E1 portfolio-level combo of stable-7 book vs lowvol-top-15 book
#   E2 low-vol filter: top-20 by stable-7 signal, keep the 15 lowest-vol
print("E: risk-adjuster channels...", file=sys.stderr, flush=True)
net7 = net_constrained(sig_base)
net_lv = net_constrained(cst["lowvol40"])
E = {"lowvol40_top15_sleeve": stats_constrained(net_lv)}
for wa in (0.7, 0.5):
    E[f"combo_{int(wa*100)}_{int((1-wa)*100)}_s7_lv"] = stats_constrained(wa * net7 + (1 - wa) * net_lv)
sig_filt = pd.DataFrame(np.nan, index=days, columns=close.columns)
for i, t in enumerate(days):
    if i % REBAL != 0:
        continue
    rowv = sig_base.loc[t].dropna()
    if len(rowv) < 20:
        continue
    top20 = rowv.nlargest(20).index
    keep15 = cst["lowvol40"].loc[t, top20].nlargest(15).index
    sig_filt.loc[t, keep15] = sig_base.loc[t, keep15]
E["stable7_top20_lowvol15_filter"] = stats_constrained(net_constrained(sig_filt))

out = {"description": "CST-quant claims tested on CSI300 under our gates (incl. gate blind-spot checks)",
       "A_daily_ic": claim_ic, "B_gate2_corr": corrs,
       "C_gate3_blends": blends, "D_trio_sleeves": sleeves,
       "E_risk_adjuster_channels": E}
json.dump(out, open(DATA / "csi300_cst_practice.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

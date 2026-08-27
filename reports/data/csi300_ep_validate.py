"""EP blend validation (2026-08-21): is stable7+EP real or a single-test fluke?

csi300_cst_practice.py found stable7+ep = 31.9%/1.14 vs baseline 29.4%/1.04 —
the first positive gate-3 result in two days, from a factor BELOW the 0.02 IC
bar. Validate before believing (monthly-track lesson):
  V1 ex-2020 and ex-bull(2024-25) Sharpe
  V2 fixed-weight sensitivity: 10% and 20% EP (not IR-weighted) — does the
     gain survive without letting the IR mechanism lean on EP?
  V3 EP weight share in the IR blend (how much is EP doing?)
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

fac7 = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
ep = zscore(reg.compute("fund_earnings_yield", panel))

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

def build_signal(mode):
    sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
    epw_log = []
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(pd.Series([fac7[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
        if mode == "ir_blend":
            irs["ep"] = ir_of(pd.Series([ep.loc[t].corr(fwd_clean.loc[t]) for t in win], index=win))
            wsum = sum(abs(v) for v in irs.values()) or 1.0
            wts = {a: v / wsum for a, v in irs.items()}
            epw_log.append(abs(wts.get("ep", 0)))
            blk_f = {**fac7, "ep": ep}
            sig.loc[days[start:start + RETRAIN]] = sum(blk_f[a].loc[days[start:start + RETRAIN]] * wts[a] for a in wts)
        elif mode.startswith("fixed_"):
            w_ep = float(mode.split("_")[1]) / 100
            wsum = sum(abs(v) for v in irs.values()) or 1.0
            wts = {a: v / wsum * (1 - w_ep) for a, v in irs.items()}
            sig.loc[days[start:start + RETRAIN]] = (
                sum(fac7[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7) +
                ep.loc[days[start:start + RETRAIN]] * w_ep)
        else:  # baseline
            wsum = sum(abs(v) for v in irs.values()) or 1.0
            wts = {a: v / wsum for a, v in irs.items()}
            sig.loc[days[start:start + RETRAIN]] = sum(fac7[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)
    return sig, epw_log

def net_of(sig):
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
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

def full_stats(net, label):
    net = net[net.index >= OOS_START]
    eq = (1 + net).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    sh = cagr / (net.std() * np.sqrt(252)) if net.std() else None
    def sh_of(sub):
        if len(sub) < 50 or sub.std() == 0: return None
        e = (1 + sub).cumprod()
        y = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
        c = float(e.iloc[-1] ** (1 / y) - 1)
        return round(c / (sub.std() * np.sqrt(252)), 2)
    return {"label": label, "cagr_pct": round(cagr * 100, 1), "sharpe": round(sh, 2),
            "sharpe_ex2020": sh_of(net[net.index.year != 2020]),
            "sharpe_ex_bull2425": sh_of(net[~net.index.year.isin([2024, 2025])]),
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

results = {}
ep_shares = {}
for mode in ["baseline", "ir_blend", "fixed_10", "fixed_20"]:
    print(mode, file=sys.stderr, flush=True)
    sig, epw = build_signal(mode)
    results[mode] = full_stats(net_of(sig), mode)
    if epw:
        ep_shares[mode] = {"mean_abs_w": round(float(np.mean(epw)), 3),
                           "max_abs_w": round(float(np.max(epw)), 3)}
out = {"results": results, "ep_weight_in_ir_blend": ep_shares}
json.dump(out, open(DATA / "csi300_ep_validate.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

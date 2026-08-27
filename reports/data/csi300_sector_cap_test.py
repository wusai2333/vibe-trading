"""Sector-cap incremental test for stable-7 (2026-08-19, user approved).

Motivation: 08-19 crash diagnose found all 7 daily-excess <=-4% days in
7.5y OOS are 2026 tech-concentration reversals (14/20 IT+telecom = 97.2%
percentile). The model has NO sector cap. This test applies per-sector
seat caps at selection time and runs both engines (decision = constrained).

Variants share the IDENTICAL stable-7 signal (block-wise 252d IR weights,
63d retrain, rolling(10,min_periods=6)); only selection differs:
  baseline: Top-15, no cap   cap8/cap6/cap5: max N seats per sector,
  filling from further down the ranked list. Locked positions (limit-down/
suspended, cannot sell) keep their seats regardless of cap.

Caveat: sector membership = current csindex cache applied historically
(constituents change quarterly; approximation).
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
sym_sec = {s: sec_map.get(s.split(".")[0], "其他") for s in close.columns}

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

sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
print("building signal...", file=sys.stderr, flush=True)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + RETRAIN]
    sig.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())

pool_ret = fwd_clean.mean(axis=1)

def capped_top(rowv, cap):
    picked, cnt = [], {}
    for s in rowv.sort_values(ascending=False).index:
        if len(picked) >= TOP_N: break
        sec = sym_sec[s]
        if cnt.get(sec, 0) >= cap: continue
        picked.append(s); cnt[sec] = cnt.get(sec, 0) + 1
    return picked

def tail_stats(net, ex):
    oos_ex = ex[ex.index >= OOS_START]
    worst = oos_ex.nsmallest(5)
    return {"worst_daily_net_pct": round(float(net[oos_ex.index].min()) * 100, 2),
            "days_ex_le_-4pct": int((oos_ex <= -0.04).sum()),
            "days_ex_le_-3pct": int((oos_ex <= -0.03).sum()),
            "worst_ex_days": {str(t.date()): round(float(v) * 100, 2) for t, v in worst.items()},
            "ex_2026_mean_pct": round(float(oos_ex[oos_ex.index.year == 2026].mean()) * 100, 3),
            "ex_2026_le_-4pct": int((oos_ex[oos_ex.index.year == 2026] <= -0.04).sum())}

def stats_from(net, label):
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
              for y in sorted(set(eq.index.year))}
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": yearly}

def run(sig, cap):
    # unconstrained, cap-aware selection
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    last = None
    for i, t in enumerate(sig.index):
        if last is None or i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                last = set(capped_top(rowv, cap))
        if last:
            w.loc[t, list(last)] = 1.0 / TOP_N
    gross = (w * fwd_clean).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net_u = gross - turn * 2 * COST
    # constrained, cap-aware selection
    wc = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    ev = {"buy_blocked_limit_up": 0, "buy_blocked_suspended": 0,
          "sell_locked_limit_down": 0, "sell_locked_suspended": 0, "cap_skips": 0}
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(capped_top(rowv, cap))
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s]: locked.add(s); ev["sell_locked_suspended"] += 1
                    elif limit_down.at[t, s]: locked.add(s); ev["sell_locked_limit_down"] += 1
                cnt = {}
                for s in keep | locked:
                    cnt[sym_sec[s]] = cnt.get(sym_sec[s], 0) + 1
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held: continue
                    sec = sym_sec[s]
                    if cnt.get(sec, 0) >= cap:
                        ev["cap_skips"] += 1; continue
                    if not tradable.at[t, s]: ev["buy_blocked_suspended"] += 1
                    elif limit_up.at[t, s]: ev["buy_blocked_limit_up"] += 1
                    else:
                        buys.append(s); cnt[sec] = cnt.get(sec, 0) + 1
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            wc.loc[t, list(held)] = 1.0 / n
    grossc = (wc.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turnc = (wc.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net_c = grossc - turnc * 2 * COST
    oos_idx = days[days >= OOS_START]
    return {"unconstrained": stats_from(net_u, f"unc cap{cap}"),
            "constrained": stats_from(net_c, f"con cap{cap}"),
            "constraint_events": ev,
            "tail_unconstrained": tail_stats(net_u, (net_u - pool_ret.shift(1).fillna(0))[oos_idx]),
            "tail_constrained": tail_stats(net_c, (net_c - pool_ret.shift(1).fillna(0))[oos_idx])}

out = {}
for name, cap in {"baseline": TOP_N, "cap8": 8, "cap6": 6, "cap5": 5}.items():
    print(f"running {name}...", file=sys.stderr, flush=True)
    out[name] = run(sig, cap)
    c = out[name]["constrained"]; u = out[name]["unconstrained"]
    tc = out[name]["tail_constrained"]
    print(f"  {name}: unc {u['cagr_pct']}% / con {c['cagr_pct']}% Sharpe {c['sharpe']} MaxDD {c['max_dd_pct']}% "
          f"tail<=-4%: {tc['days_ex_le_-4pct']} (2026: {tc['ex_2026_le_-4pct']}) cap_skips {out[name]['constraint_events']['cap_skips']}")

json.dump(out, open(DATA / "csi300_sector_cap_test.json", "w"), ensure_ascii=False, indent=1)
print(f"SAVED {DATA / 'csi300_sector_cap_test.json'}", file=sys.stderr)
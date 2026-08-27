"""Stable-5 OOS backtest with realistic trading constraints.

Rebuilds the production rolling-weight backtest (lost csi300_rolling_weights.py)
and adds A-share microstructure constraints:

  * limit-up  : cannot BUY a stock sealed at the upper limit
  * limit-down: cannot SELL a stock sealed at the lower limit (held over)
  * suspension: close NaN / zero volume -> no trading at all; the position is
                frozen (0 return) until it resumes

Limit thresholds: main board 10%; ChiNext (300/301) 20% since the 2020-08-24
registration reform (10% before); STAR (688) always 20%. A stock counts as
"sealed" when the day's return is within 0.2pp of its limit.

Runs on the scrubbed + Sina-repaired panel (panel_scrub.py / panel_repair.py),
plus a final return guard that zeroes residual impossible one-day moves (see
below). Both the unconstrained baseline (original engine, verbatim mechanics)
and the constrained variant run on the SAME signal, Top-15 equal weight,
5-day rebalance, 10bps one-way cost.

Signal (identical to production, no future information):
  5 factors, cross-sectional z-score, rolling(10, min_periods=6) smoothing,
  trailing-252d IC->IR weights, retrained every 63 trading days, applied to
  the following 63 days.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
volume = panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)          # same convention as production
ret = close.pct_change()

STABLE_IDS = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

# ---- tradability masks ----
tradable = close.notna() & volume.fillna(0).gt(0)
def limit_pct() -> pd.DataFrame:
    lim = pd.DataFrame(0.10, index=days, columns=close.columns)
    star = [c for c in close.columns if c.startswith("688")]
    gem = [c for c in close.columns if c.startswith("30")]
    if star:
        lim[star] = 0.20
    if gem:
        reform = pd.Timestamp("2020-08-24")
        lim.loc[days >= reform, gem] = 0.20
    return lim
lim = limit_pct()

# ---- final data guard ----
# After the level scrub + Sina repair, a few splices still leave persistent
# jumps the level check cannot see (e.g. 601088.SH 2018-07-09 +54%). An
# A-share cannot move beyond its daily limit in one session; the exception is
# resuming from a LONG suspension (first day back without price limits, e.g.
# 000792.SZ +306% after its bankruptcy-restructuring resumption). Zero
# non-post-suspension anomalous returns in the P&L accounting and treat such
# days as untradable.
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd_clean = fwd.mask(anomalous.shift(-1).fillna(False), 0.0)
print(f"return guard: zeroing {int(anomalous.sum().sum())} anomalous return cells",
      file=sys.stderr)

tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()

def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", file=sys.stderr)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE_IDS}

def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

# ---- rolling weights: train on the 252 days ending the day before `start` ----
signal = pd.DataFrame(np.nan, index=days, columns=close.columns)
weight_log = []
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]              # fwd known before `start`
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win))
           for a in STABLE_IDS}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    blk = days[start:start + RETRAIN]
    signal.loc[blk] = sum(fac[a].loc[blk] * wts[a] for a in STABLE_IDS)
    weight_log.append({"applied": str(blk[0].date()), **{k: round(v, 3) for k, v in wts.items()}})
print(f"rolling weights: {len(weight_log)} blocks", file=sys.stderr)

def stats_from(net: pd.Series, w: pd.DataFrame, label: str) -> dict:
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    yearly = {str(y): round(float((eq[eq.index.year == y].iloc[-1] /
                                    eq[eq.index.year == y].iloc[0] - 1) * 100), 1)
              for y in sorted(set(eq.index.year))}
    return {
        "label": label,
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(cagr / vol, 2) if vol > 0 else None,
        "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
        "yearly_pct": yearly,
    }

def backtest_baseline(sig: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Original engine, verbatim mechanics (no constraints)."""
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
    net = gross - turn * 2 * COST
    return stats_from(net, w, "baseline_unconstrained"), w

def backtest_constrained(sig: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Position simulator with limit/suspension constraints. Signal day = trade
    day (same convention as the baseline); sealed/suspended names can't trade."""
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held: set = set()
    ev = {"rebalances": 0, "buy_blocked_limit_up": 0, "buy_blocked_suspended": 0,
          "sell_locked_limit_down": 0, "sell_locked_suspended": 0,
          "over_hold_days": 0, "under_invested_days": 0}
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                ev["rebalances"] += 1
                desired = list(rowv.nlargest(TOP_N).index)  # ranked
                dset = set(desired)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s]:
                        locked.add(s); ev["sell_locked_suspended"] += 1
                    elif limit_down.at[t, s]:
                        locked.add(s); ev["sell_locked_limit_down"] += 1
                # fill slots from the ranking (past rank 15 if needed),
                # skipping suspended / limit-up names
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N:
                        break
                    if s in held:
                        continue
                    if not tradable.at[t, s]:
                        ev["buy_blocked_suspended"] += 1
                    elif limit_up.at[t, s]:
                        ev["buy_blocked_limit_up"] += 1
                    else:
                        buys.append(s)
                held = keep | locked | set(buys)
        if held:
            n = max(len(held), TOP_N)
            w.loc[t, list(held)] = 1.0 / n
            if len(held) > TOP_N:
                ev["over_hold_days"] += 1
            if len(held) < TOP_N:
                ev["under_invested_days"] += 1
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    return stats_from(net, w, "constrained_limits_suspension"), w, ev

print("running baseline...", file=sys.stderr)
base_stats, w_base = backtest_baseline(signal)
print("running constrained...", file=sys.stderr)
con_stats, w_con, events = backtest_constrained(signal)

out = {
    "description": "stable-5 rolling-weight OOS backtest on the scrubbed+Sina-repaired panel; baseline vs limit/suspension constraints",
    "signal": {"factors": STABLE_IDS, "train_days": TRAIN, "retrain_days": RETRAIN,
               "top_n": TOP_N, "rebalance_days": REBAL, "cost_one_way": COST,
               "oos_start": str(OOS_START.date()), "panel_end": str(days[-1].date())},
    "results": {"baseline": base_stats, "constrained": con_stats},
    "constraint_events": events,
    "anomalous_returns_zeroed": int(anomalous.sum().sum()),
    "weight_blocks_first3": weight_log[:3],
}
json.dump(out, open(DATA / "csi300_constrained_backtest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out["results"], ensure_ascii=False, indent=1))
print(json.dumps(events, ensure_ascii=False, indent=1))
print(f"SAVED {DATA / 'csi300_constrained_backtest.json'}", file=sys.stderr)

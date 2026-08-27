"""Short-leg conversion via reverse exclusion (Haitong 2016 report #16 method).

Their finding: most A-share factors earn >70% of their IC from the short leg;
without shorting, EXCLUDE the factor's worst names from the universe (逆向剔除)
and hold the rest — captures part of the short leg as avoided losses.

Test on our best non-executable signal: ZZ1000 LGBM monthly (LS ex-bull 1.55,
long-only ex-bull 0.13). Question: can reverse exclusion monetize part of the
short leg without securities lending?

Variants (from saved preds, no retrain):
  EX100: exclude bottom-100 predicted, EW hold the rest (~900 names)
  EX200: exclude bottom-200 (~800 names)
  T50X100: exclude bottom-100, then top-50 by score (exclusion + ranking,
           different from plain top-50 only if bottom names would be picked —
           sanity check, should ~= plain top-50)
Costs: turnover-based 10bps one-way.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
art = pickle.load(open(DATA / "csi1000_lgbm_monthly.pkl", "rb"))
pred = art["pred"]
panel = pickle.load(open(DATA / "csi1000_panel.pkl", "rb"))
close = panel["close"]
days = close.index
ret = close.pct_change()
fwd1 = close.pct_change().shift(-1)
OOS_START = pred["date"].min()

tradable = close.notna() & panel["volume"].fillna(0).gt(0)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd1_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous

COST = 0.001
rebal_dates = sorted(pred["date"].unique())
day_pos = {d: i for i, d in enumerate(days)}

def sleeve(excl_n, top_n=None):
    w = pd.DataFrame(0.0, index=days, columns=close.columns)
    held = {}
    for bi, d in enumerate(rebal_dates):
        grp = pred[pred["date"] == d].dropna(subset=["pred"])
        if len(grp) < 300: continue
        if excl_n:
            bottom = set(grp.nsmallest(excl_n, "pred")["stock"])
            pool = grp[~grp["stock"].isin(bottom)]
        else:
            pool = grp
        if top_n:
            pick = pool.nlargest(top_n, "pred")["stock"].tolist()
        else:
            pick = pool["stock"].tolist()
        pick = [s for s in pick if s in close.columns and bool(tradable.at[d, s])]
        if not pick: continue
        end = rebal_dates[bi + 1] if bi + 1 < len(rebal_dates) else days[-1]
        w.loc[d:end, pick] = 1.0 / len(pick)
    gross = (w * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    net = net[net.index >= OOS_START]
    eq = (1 + net).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    sh = cagr / (net.std() * np.sqrt(252)) if net.std() else None
    exb = net[~net.index.year.isin([2024, 2025])]
    shx = None
    if len(exb) > 50 and exb.std() > 0:
        eqx = (1 + exb).cumprod()
        yx = max((eqx.index[-1] - eqx.index[0]).days / 365.25, 1e-9)
        shx = round(float(eqx.iloc[-1] ** (1 / yx) - 1) / (exb.std() * np.sqrt(252)), 2)
    to = float(turn[net.index].mean())
    return {"ann_pct": round(cagr * 100, 1), "sharpe": round(sh, 2) if sh else None,
            "sharpe_ex_bull2425": shx,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "monthly_turnover": round(to, 3),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

results = {}
for name, ex, tn in [("EX100_broad", 100, None), ("EX200_broad", 200, None),
                     ("T50X100_sanity", 100, 50), ("T50_plain", 0, 50)]:
    print(name, file=sys.stderr, flush=True)
    results[name] = sleeve(ex, tn)
json.dump(results, open(DATA / "csi1000_shortleg_exclusion.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))

"""easonZC factorlab diagnostics on our bench (2026-08-21).

Their framework's borrowable diagnostics, tested on our known alive/dead
spectrum to see if they separate signal from noise (and catch what IC misses):

  D1 quantile monotonicity: 10-quantile x fwd5d mean returns, Spearman rho
     (catches non-monotonic factors IC can't — CST's F5 failure mode)
  D2 Fama-MacBeth beta + Newey-West t (daily cs regressions, univariate)
  D3 rank autocorrelation lag-1 (stability/turnover proxy)
  D4 IC decay across horizons 1/5/21d (horizon-mismatch detector)

Subjects: stable-7 factors + EP (alive side) + mom40/roe (dead side).
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()
ret = close.pct_change()
fwd = {h: close.pct_change(h).shift(-h) for h in (1, 5, 21)}

OOS_START = pd.Timestamp("2019-01-01")
oos = days[days >= OOS_START]

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", file=sys.stderr, flush=True)
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
factors = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
factors["ep"] = zscore(reg.compute("fund_earnings_yield", panel))
factors["mom40"] = zscore(close.shift(21) / close.shift(61) - 1)
factors["roe"] = zscore(panel["fund:roe"])

def nw_t(beta: pd.Series, lags=None):
    b = beta.dropna()
    n = len(b)
    if n < 60: return None
    l = lags or int(np.floor(4 * (n / 100) ** (2 / 9)))
    mu = b.mean()
    x = b - mu
    gam0 = float((x * x).sum() / n)
    var = gam0
    for k in range(1, l + 1):
        gamk = float((x.iloc[k:] * x.iloc[:-k]).sum() / n)
        var += 2 * (1 - k / (l + 1)) * gamk
    se = np.sqrt(max(var, 1e-12) / n)
    return float(mu / se) if se > 0 else None

rows = {}
for name, f in factors.items():
    print(name, file=sys.stderr, flush=True)
    row = {}
    # D1 quantile monotonicity (10 quantiles, fwd5d, OOS)
    q_ret = {q: [] for q in range(10)}
    for t in oos[::5]:  # every 5d to reduce overlap
        fv = f.loc[t].dropna()
        rv = fwd[5].loc[t].reindex(fv.index).dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < 100: continue
        q = pd.qcut(fv[common].rank(method="first"), 10, labels=False)
        for qi in range(10):
            sel = common[q.values == qi]
            if len(sel): q_ret[qi].append(float(rv[sel].mean()))
    prof = [np.mean(v) if v else np.nan for v in q_ret.values()]
    row["quantile_profile_5d"] = [round(x * 100, 3) for x in prof]
    valid = [(i, p) for i, p in enumerate(prof) if np.isfinite(p)]
    row["monotonicity_rho"] = round(float(spearmanr([i for i, _ in valid], [p for _, p in valid]).correlation), 3) if len(valid) >= 8 else None
    row["q10_minus_q1_5d_pct"] = round((prof[9] - prof[0]) * 100, 3) if np.isfinite(prof[9]) and np.isfinite(prof[0]) else None
    # D2 Fama-MacBeth (daily cs univariate regression on fwd1)
    betas = []
    for t in oos:
        fv = f.loc[t].dropna()
        rv = fwd[1].loc[t].reindex(fv.index).dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < 100: continue
        x = fv[common].values
        xc = x - x.mean()
        denom = (xc * xc).sum()
        if denom <= 0: continue
        betas.append(float((xc * (rv[common].values - rv[common].mean())).sum() / denom))
    bs = pd.Series(betas)
    row["fmb_beta_mean"] = round(float(bs.mean()) * 100, 4) if len(bs) else None  # %/day per z
    row["fmb_nw_t"] = round(nw_t(bs), 2) if len(bs) else None
    # D3 rank autocorrelation lag-1
    rks = f.rank(axis=1)
    acs = []
    for i in range(1, len(oos)):
        t0_, t1 = oos[i - 1], oos[i]
        a, b = rks.loc[t0_], rks.loc[t1]
        common = a.dropna().index.intersection(b.dropna().index)
        if len(common) > 100:
            acs.append(float(a[common].corr(b[common], method="spearman")))
    row["rank_autocorr_lag1"] = round(float(np.mean(acs)), 3) if acs else None
    # D4 IC decay
    for h in (1, 5, 21):
        ics = []
        step = max(h, 1)
        for t in oos[::step]:
            fv = f.loc[t].dropna()
            rv = fwd[h].loc[t].reindex(fv.index).dropna()
            common = fv.index.intersection(rv.index)
            if len(common) > 100:
                c = float(fv[common].corr(rv[common], method="spearman"))
                if np.isfinite(c): ics.append(c)
        row[f"ic_h{h}"] = round(float(np.mean(ics)), 4) if ics else None
    rows[name] = row

json.dump(rows, open(DATA / "csi300_factorlab_diagnostics.json", "w"), ensure_ascii=False, indent=1)
hdr = f"{'factor':16s} {'mono_rho':>8s} {'q10-q1%':>8s} {'fmb_t':>6s} {'rk_ac1':>6s} {'ic_h1':>7s} {'ic_h5':>7s} {'ic_h21':>7s}"
print(hdr)
for name, r in rows.items():
    print(f"{name:16s} {str(r['monotonicity_rho']):>8s} {str(r['q10_minus_q1_5d_pct']):>8s} "
          f"{str(r['fmb_nw_t']):>6s} {str(r['rank_autocorr_lag1']):>6s} "
          f"{str(r['ic_h1']):>7s} {str(r['ic_h5']):>7s} {str(r['ic_h21']):>7s}")

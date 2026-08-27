"""stable-7 crash diagnose (2026-08-19): factor death or risk gap?

08-19 Top20 tracking excess was -6.32% (worst on record). Two questions:
  A. Factor death? -> daily IC series of the blend + per-factor IC, recent
     window vs history; crash-day within-sector IC (did rank fail inside
     tech, or was it pure sector beta?).
  B. Risk gap? -> historical daily Top20-vs-pool excess distribution
     (same protocol as daily tracking): how often does a day <= -6.32%
     happen; sector concentration of Top20 over history vs today's 14/20.
Read-only; prints JSON-ish summary. Reuses csi300_limit_vol_test mechanics.
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
TRAIN, RETRAIN = 252, 63
OOS = pd.Timestamp("2019-01-01")

# tradability + guard (same as production scripts)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd_clean = fwd.mask(anomalous.shift(-1).fillna(False), 0.0)

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

# block-wise IR weights -> blend signal (production mechanics)
sig = pd.DataFrame(np.nan, index=days, columns=close.columns)
print("building signal...", file=sys.stderr, flush=True)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + RETRAIN]
    sig.loc[blk] = sum(fac[a].loc[blk] * (v / wsum) for a, v in irs.items())

# ---- A1: blend daily IC (signal rank vs next-day return) ----
ic_blend = sig.corrwith(fwd_clean, axis=1)
oos = ic_blend[ic_blend.index >= OOS].dropna()
recent = oos.tail(21)  # last month incl. crash day
print("\n=== A1: blend daily IC ===")
print(f"OOS mean {oos.mean():.4f} std {oos.std():.4f}  hit-rate {(oos>0).mean():.1%}")
print(f"recent 21d mean {recent.mean():.4f}  (excl last day {recent.iloc[:-1].mean():.4f})")
print("last 6 days IC:")
for t, v in oos.tail(6).items():
    print(f"  {t.date()}  {v:+.4f}")

# ---- A2: per-factor IC, recent-20d-excl-crash vs history ----
print("\n=== A2: per-factor IC (recent 20d excl crash vs OOS mean) ===")
for a in STABLE7:
    ica = fac[a].corrwith(fwd_clean, axis=1)
    ica = ica[ica.index >= OOS].dropna()
    r20 = ica.iloc[-21:-1]
    print(f"  {a:15s} hist {ica.mean():+.4f}  recent20 {r20.mean():+.4f}  crash-day {ica.iloc[-1]:+.4f}")

# ---- A3: crash-day within-sector IC ----
sec_map = json.load(open(DATA / "stock2sector_cache.json"))
sym_sec = pd.Series({s: sec_map.get(s.split(".")[0], "其他") for s in close.columns})
t0, t1 = days[-2], days[-1]
s_row, r_row = sig.loc[t0], fwd_clean.loc[t0]
ok = s_row.notna() & r_row.notna()
tech = sym_sec.isin(["信息技术", "电信服务"])
def ic(mask):
    m = ok & mask
    return float(np.corrcoef(s_row[m], r_row[m])[0, 1]) if m.sum() > 10 else float("nan")
print("\n=== A3: crash day (sig@08-18 vs ret@08-19) ===")
print(f"  whole cross-section IC {ic(ok):+.3f}  (n={int(ok.sum())})")
print(f"  within tech+telecom    IC {ic(tech):+.3f}  (n={int((ok&tech).sum())})")
print(f"  within non-tech        IC {ic(~tech):+.3f}  (n={int((ok&~tech).sum())})")
pool_tech = r_row[ok & tech].mean(); pool_rest = r_row[ok & ~tech].mean()
print(f"  pool return: tech {pool_tech:+.2%} vs non-tech {pool_rest:+.2%} -> sector beta gap {pool_tech-pool_rest:+.2%}")

# ---- B1: historical daily Top20-vs-pool excess (tracking protocol) ----
print("\n=== B1: Top20 daily excess distribution (OOS) ===")
S = sig[ok.index].to_numpy() if False else None
sigv = sig.values; fwdv = fwd_clean.values
oos_pos = days.get_indexer(pd.DatetimeIndex([d for d in days if d >= OOS]))
excess, exdays = [], []
pool_mean = fwdv.mean(axis=1)
for i in range(len(days)):
    row = sigv[i]
    valid = ~np.isnan(row) & ~np.isnan(fwdv[i])
    if valid.sum() < 60 or days[i] < OOS: continue
    top = np.argpartition(np.where(valid, row, -np.inf), -20)[-20:]
    excess.append(fwdv[i][top].mean() - pool_mean[i])
    exdays.append(days[i])
ex = pd.Series(excess, index=exdays)
print(f"days {len(ex)}  mean {ex.mean():+.3%}  std {ex.std():.3%}")
print(f"worst 8 days ever:")
for t, v in ex.nsmallest(8).items():
    print(f"  {t.date()}  {v:+.2%}")
worse = (ex <= -0.0632).sum()
print(f"days <= -6.32%: {worse} / {len(ex)}  ({(ex <= -0.0632).mean():.2%})")
print(f"days <= -4%: {(ex <= -0.04).sum()}  <= -3%: {(ex <= -0.03).sum()}")
print(f"2026 worst before 08-19: {ex[ex.index < days[-1]].tail(252).min():+.2%}")

# ---- B2: Top20 sector concentration history vs today ----
print("\n=== B2: Top20 concentration ===")
shares, maxsec = [], []
for i in range(len(days)):
    if days[i] < OOS: continue
    row = sigv[i]
    valid = ~np.isnan(row)
    if valid.sum() < 60: continue
    top = np.argpartition(np.where(valid, row, -np.inf), -20)[-20:]
    secs = sym_sec.iloc[top].values
    u, c = np.unique(secs, return_counts=True)
    maxsec.append(c.max() / 20)
    shares.append((secs == "信息技术").sum() + (secs == "电信服务").sum())
ms = pd.Series(maxsec, index=exdays[:len(maxsec)] if len(maxsec)==len(exdays) else None)
sh = pd.Series(shares)
print(f"max single-sector seat share: mean {np.mean(maxsec):.1%}  p95 {np.quantile(maxsec,.95):.1%}  max {np.max(maxsec):.1%}")
print(f"IT+telecom seats in Top20: mean {sh.mean():.1f}  p95 {sh.quantile(.95):.0f}  max {sh.max():.0f}")
print(f"08-18 signal (crashed portfolio): 14/20 -> percentile {float((sh <= 14).mean()):.1%}")
print(f"08-19 signal (today): 12/20 -> percentile {float((sh <= 12).mean()):.1%}")

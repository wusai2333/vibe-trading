"""Vectorized cross-sectional Spearman gate for CSI500 session candidates.

Daily Spearman = Pearson on per-day ranks; computed fully vectorized
(no per-day scipy calls). Averages over all days with >=100 joint-valid names.
"""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent
vals = pickle.load(open(DATA / "csi500_factor_cache.pkl", "rb"))
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
CANDS = ["session_onin20", "session_on20", "session_on5"]


def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def avg_cs_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ra, rb = a.rank(axis=1), b.rank(axis=1)
    m = a.notna() & b.notna()
    n = m.sum(axis=1)
    keep = n >= 100
    ra, rb, n = ra[keep], rb[keep], n[keep]
    ma = ra.sum(axis=1) / n
    mb = rb.sum(axis=1) / n
    da = (ra.sub(ma, axis=0)).where(m[keep])
    db = (rb.sub(mb, axis=0)).where(m[keep])
    num = (da * db).sum(axis=1)
    den = np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))
    return float((num / den).dropna().mean())


zv = {k: zscore(v) for k, v in vals.items()}
print("candidates vs stable-7 (avg cross-sectional Spearman, 2018-2026):")
for c in CANDS:
    cors = {a: avg_cs_spearman(zv[c], zv[a]) for a in STABLE7}
    top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"  {c:16s} " + ", ".join(f"{a}={r:+.3f}" for a, r in top))
print("candidates vs each other:")
for i, a in enumerate(CANDS):
    for b in CANDS[i + 1:]:
        print(f"  {a} vs {b}: {avg_cs_spearman(zv[a], zv[b]):+.3f}")

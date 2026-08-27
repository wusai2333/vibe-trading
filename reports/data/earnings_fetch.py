"""Fetch earnings preview (业绩预告) + express (业绩快报) per quarter, resumable.

Sources: akshare stock_yjyg_em / stock_yjkb_em (eastmoney, probed OK 2026-08-20).
Raw per-quarter frames cached under reports/data/earnings_cache/.
"""
import os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import akshare as ak

CACHE = Path("reports/data/earnings_cache")
CACHE.mkdir(exist_ok=True)

quarters = pd.date_range("2018-03-31", "2026-06-30", freq="QE")

def fetch(fn, tag, q):
    key = tag + "_" + str(q.year) + "Q" + str((q.month - 1) // 3 + 1) + ".pkl"
    p = CACHE / key
    if p.exists():
        return "cached"
    ds = q.strftime("%Y%m%d")
    df = None
    for k in range(3):
        try:
            df = fn(date=ds)
            break
        except Exception as e:
            if k == 2:
                print("  GIVEUP " + key + ": " + type(e).__name__ + " " + str(e)[:80], flush=True)
            time.sleep(4)
    tmp = CACHE / (".tmp_" + key)
    pickle.dump(df, open(tmp, "wb"))
    os.replace(tmp, p)
    return "ok " + str(0 if df is None else len(df))

for q in quarters:
    r1 = fetch(ak.stock_yjyg_em, "yg", q)
    time.sleep(0.4)
    r2 = fetch(ak.stock_yjkb_em, "kb", q)
    time.sleep(0.4)
    print(str(q.date()) + ": yg " + r1 + ", kb " + r2, flush=True)
print("FETCH DONE")

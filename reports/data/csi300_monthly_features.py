"""ZZ1000 monthly feature dataset from the FULL zoo (streaming, resumable).

Rows kept = month-end (last trading day) cross-sections only, so the daily
factor panels are never held in memory. Per-factor try/except: alphas needing
missing columns (amount/sector) are skipped and counted. Partial cache saved
atomically every 25 factors (env kills long jobs; resume loads the cache).

Output: csi300_monthly_features.pkl = {"feats": {alpha: DataFrame(rebal x stocks)},
        "rebal": [...], "skipped": [...]}
"""
import sys, os, time, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
from itertools import groupby
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent
OUT = DATA / "csi300_monthly_features.pkl"
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
panel.pop("_meta", None)
close = panel["close"]
dates = [str(d.date()).replace("-", "") for d in close.index]

# month-end rebalance dates (their build_monthly rule)
rebal = [list(g)[-1] for _, g in groupby(close.index, key=lambda d: (d.year, d.month))]
rebal_idx = pd.DatetimeIndex(rebal)
print(f"{len(rebal_idx)} month-end dates: {rebal_idx[0].date()} -> {rebal_idx[-1].date()}", flush=True)

from src.factors.registry import get_default_registry
reg = get_default_registry()
ids = sorted(reg.list())
print(f"zoo size: {len(ids)}", flush=True)

cache, skipped = {}, []
if OUT.exists():
    old = pickle.load(open(OUT, "rb"))
    cache, skipped = old["feats"], old.get("skipped", [])
    print(f"resuming: {len(cache)} cached, {len(skipped)} skipped", flush=True)

def save():
    tmp = OUT.with_suffix(".tmp")
    pickle.dump({"feats": cache, "rebal": list(rebal_idx), "skipped": skipped}, open(tmp, "wb"))
    os.replace(tmp, OUT)

t0 = time.time()
todo = [a for a in ids if a not in cache and a not in skipped]
print(f"todo: {len(todo)}", flush=True)
for i, aid in enumerate(todo):
    try:
        f = reg.compute(aid, panel)
        cache[aid] = f.loc[rebal_idx]
    except Exception as e:
        skipped.append(aid)
        if len(skipped) <= 5:
            print(f"  skip {aid}: {type(e).__name__} {str(e)[:80]}", flush=True)
    if (i + 1) % 25 == 0:
        save()
        el = time.time() - t0
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i+1}/{len(todo)}] ok={len(cache)} skip={len(skipped)} "
              f"elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

save()
print(f"DONE ok={len(cache)} skipped={len(skipped)} wall={(time.time()-t0)/60:.1f}m", flush=True)

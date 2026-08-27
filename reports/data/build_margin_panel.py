"""Build per-stock margin-trading (两融) PIT panel from SZSE/SSE official sites.

Crash-proof edition: raw per-date frames go into append-only part files
(margin_cache_parts/part_XXXX.pkl, 50 dates each, written once via tmp+
os.replace — a killed process can never corrupt written parts). Resumable.

Usage: python build_margin_panel.py --limit N   (fetch at most N new dates)
When all dates are cached, assembles margin_panel.pkl (wide panels).

PIT NOTE: margin data for trading day d is published after the d close.
The panel stores RAW values at their credit date; factor modules must
shift(1) so that signal at t only uses margin data up to t-1.
"""
import os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import akshare as ak

DATA = Path(__file__).resolve().parent
PARTS = DATA / "margin_cache_parts"
PART_SIZE = 50
OUT = DATA / "margin_panel.pkl"
START = pd.Timestamp("2018-01-01")
PARTS.mkdir(exist_ok=True)

panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
days = [d for d in panel["close"].index if d >= START]

cache = {}
for pf in sorted(PARTS.glob("part_*.pkl")):
    cache.update(pickle.load(open(pf, "rb")))
print(f"resumed: {len(cache)}/{len(days)} dates cached", file=sys.stderr, flush=True)

from concurrent.futures import ThreadPoolExecutor
_pool = ThreadPoolExecutor(max_workers=2)

def _one(fn, date_str):
    return _pool.submit(fn, date=date_str).result(timeout=45)

def fetch(fn, date_str, tries=3):
    for k in range(tries):
        try:
            return _one(fn, date_str)
        except Exception as e:
            if k == tries - 1:
                print(f"  GIVEUP {date_str}: {type(e).__name__} {str(e)[:80]}", file=sys.stderr, flush=True)
                return None
            time.sleep(5)

def save_part(buf):
    if not buf:
        return
    first = sorted(buf)[0].replace("-", "")
    target = PARTS / f"part_{first}.pkl"
    tmp = PARTS / f".tmp_{first}.pkl"
    pickle.dump(buf, open(tmp, "wb"))
    os.replace(tmp, target)

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10**9
new_done, buf = 0, {}
for d in days:
    key = str(d.date())
    if key in cache:
        continue
    if new_done >= LIMIT:
        break
    ds = d.strftime("%Y%m%d")
    sz = fetch(ak.stock_margin_detail_szse, ds)
    time.sleep(0.1)
    sh = fetch(ak.stock_margin_detail_sse, ds)
    time.sleep(0.1)
    buf[key] = (sz, sh)
    new_done += 1
    if len(buf) >= PART_SIZE:
        save_part(buf)
        cache.update(buf)
        print(f"  {len(cache)}/{len(days)} cached ({key})", file=sys.stderr, flush=True)
        buf = {}
save_part(buf)
cache.update(buf)

if len(cache) < len(days):
    print(f"CHUNK DONE: {len(cache)}/{len(days)} cached, {len(days)-len(cache)} remaining",
          file=sys.stderr, flush=True)
    sys.exit(0)

print("fetch done, assembling wide panels...", file=sys.stderr, flush=True)
rows = []
for key, (sz, sh) in sorted(cache.items()):
    d = pd.Timestamp(key)
    for df, code_col in ((sz, "证券代码"), (sh, "标的证券代码")):
        if df is None or not len(df):
            continue
        df = df.rename(columns={code_col: "code", "融资余额": "fin_balance",
                                "融资买入额": "fin_buy", "融券余量": "short_qty"})
        keep = [c for c in ["fin_balance", "fin_buy", "short_qty"] if c in df.columns]
        df = df[["code"] + keep].copy()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["sym"] = df["code"].map(lambda c: c + (".SH" if c.startswith("6") else ".SZ"))
        df["date"] = d
        rows.append(df[["date", "sym"] + keep])

long = pd.concat(rows, ignore_index=True)
for c in ["fin_balance", "fin_buy", "short_qty"]:
    long[c] = pd.to_numeric(long[c], errors="coerce")
long = long.drop_duplicates(subset=["date", "sym"], keep="last")

wide = {}
for c in ["fin_balance", "fin_buy", "short_qty"]:
    w = long.pivot(index="date", columns="sym", values=c).sort_index()
    w = w.reindex(index=pd.DatetimeIndex(days), columns=panel["close"].columns)
    wide[c] = w
    print(f"{c}: {w.shape}, non-null frac {w.notna().mean().mean():.2%}", file=sys.stderr)

pickle.dump(wide, open(OUT, "wb"))
print(f"SAVED {OUT}", file=sys.stderr)
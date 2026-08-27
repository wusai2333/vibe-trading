"""Build PIT earnings-event panel from the fetched preview/express cache.

Daily wide frames (CSI300 dates x names), information effective from the
announcement date (known at that close; signal at t predicts t -> t+1).
Outputs event_panel.pkl: surprise / type / jor / mom / kb.
"""
import pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("reports/data/earnings_cache")
OUT = Path("reports/data/event_panel.pkl")
panel = pickle.load(open("reports/data/csi300_panel.pkl", "rb"))
close = panel["close"]; open_ = panel["open"]
days = close.index
cols = close.columns

def suf(c):
    c = str(c).zfill(6)
    if c.startswith(("6", "9")): return c + ".SH"
    if c.startswith(("0", "2", "3")): return c + ".SZ"
    return None

TYPE_SCORE = {"预增": 4, "扭亏": 3, "略增": 2, "续盈": 1, "不确定": 0,
              "续亏": -1, "略减": -2, "预减": -3, "首亏": -4}

# ---- collect announcements ----
rows = []
for p in sorted(DATA.glob("yg_*.pkl")):
    qtag = p.stem.split("_", 1)[1]
    df = pickle.load(open(p, "rb"))
    if df is None or not len(df): continue
    for _, r in df.iterrows():
        sym = suf(r["股票代码"])
        if sym is None or sym not in cols: continue
        ad = pd.Timestamp(r["公告日期"])
        sur = pd.to_numeric(r.get("业绩变动幅度"), errors="coerce")
        rows.append(dict(sym=sym, period=qtag, ad=ad, surprise=float(sur) if pd.notna(sur) else np.nan,
                         tscore=TYPE_SCORE.get(str(r.get("预告类型")).strip(), 0)))
ev = pd.DataFrame(rows).sort_values(["sym", "period", "ad"])
print("previews:", len(ev), "unique names:", ev.sym.nunique())

kb_rows = []
for p in sorted(DATA.glob("kb_*.pkl")):
    qtag = p.stem.split("_", 1)[1]
    df = pickle.load(open(p, "rb"))
    if df is None or not len(df): continue
    for _, r in df.iterrows():
        sym = suf(r["股票代码"])
        if sym is None or sym not in cols: continue
        ad = pd.Timestamp(r["公告日期"])
        g = pd.to_numeric(r.get("净利润-同比增长"), errors="coerce")
        kb_rows.append(dict(sym=sym, period=qtag, ad=ad, kbg=float(g) if pd.notna(g) else np.nan))
kb = pd.DataFrame(kb_rows).sort_values(["sym", "period", "ad"])
print("express:", len(kb))

# ---- daily frames ----
sur_f = pd.DataFrame(np.nan, index=days, columns=cols)
typ_f = pd.DataFrame(np.nan, index=days, columns=cols)
mom_f = pd.DataFrame(np.nan, index=days, columns=cols)
jor_pts = []   # (sym, trade_day, jor_value)
prev_sur = {}
for _, r in ev.iterrows():
    sym, ad = r.sym, r.ad
    if ad < days[0] or pd.isna(r.surprise):
        if not pd.isna(r.surprise): pass
        else: continue
    idx = days.searchsorted(ad)
    if idx >= len(days): continue
    td = days[idx]  # effective from announcement date (or next trading day)
    sur_f.loc[td, sym] = r.surprise
    typ_f.loc[td, sym] = r.tscore
    ps = prev_sur.get(sym)
    if ps is not None:
        mom_f.loc[td, sym] = r.surprise - ps
    prev_sur[sym] = r.surprise
    if r.surprise != 0:
        o, pc = open_.loc[td, sym], close.iloc[idx - 1][sym] if idx > 0 else np.nan
        if pd.notna(o) and pd.notna(pc) and pc > 0:
            gap = o / pc - 1
            jor_pts.append((sym, td, gap * np.sign(r.surprise)))

# kb confirmation: express growth minus preview surprise of same period
kb_f = pd.DataFrame(np.nan, index=days, columns=cols)
ev_lookup = {(r.sym, r.period): r.surprise for r in ev.itertuples()}
for _, r in kb.iterrows():
    if pd.isna(r.kbg): continue
    ps = ev_lookup.get((r.sym, r.period))
    val = r.kbg - ps if ps is not None and not np.isnan(ps) else r.kbg
    idx = days.searchsorted(r.ad)
    if idx >= len(days): continue
    kb_f.loc[days[idx], r.sym] = val

sur_ff = sur_f.ffill()
typ_ff = typ_f.ffill()
mom_ff = mom_f.ffill()
kb_ff = kb_f.ffill()

# JOR: linear decay to 0 over 20 trading days
jor_f = pd.DataFrame(0.0, index=days, columns=cols)
pos = {d: i for i, d in enumerate(days)}
for sym, td, v in jor_pts:
    i0 = pos[td]
    for k in range(20):
        if i0 + k >= len(days): break
        jor_f.iloc[i0 + k, jor_f.columns.get_loc(sym)] += v * (1 - k / 20)

out = {"surprise": sur_ff, "type": typ_ff, "jor": jor_f, "mom": mom_ff, "kb": kb_ff}
for k, v in out.items():
    cov = v.notna().mean().mean()
    print(k, "coverage:", round(float(cov), 4))
pickle.dump(out, open(OUT, "wb"))
print("SAVED", OUT)

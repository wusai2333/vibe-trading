"""Authoritative clean rebuild of csi300_panel.pkl from the pre-scrub backup.

The dual-source (Tencent+Sina) qfq panel contains corrupted segments: whole
eras at 1/10th-1/50th of true level, plus persistent level jumps. Neither
source is clean everywhere — Sina has its own corrupt eras (000630.SZ 2018,
000063.SZ 2020) — so reconciliation needs an independent judge: Baostock
(adjustflag=2 qfq). Baostock's chain matches the panel's recent clean region
to <0.4% (verified on 6 stocks, 100% of days in-band), and it carries the
true level in every disputed era checked (601225 2019, 000630 2018, 300394
2018, 000063 2020, 600039 2020, 600066 2022).

Pipeline:

  1. Load the panel: the pristine backup when present, else the current
     (already build-scrubbed) panel — masked cells are then treated as holes.
  2. panel_scrub: mask panel cells whose level dislocates from centered
     rolling medians (multi-window, asymmetric thresholds, IPO exemption).
  3. Per stock fetch Sina (scrubbed with the same detector, so its garbage
     islands are masked) and Baostock qfq OHLC.
  4. reference = cleaned Sina when Sina validates against the panel's recent
     clean region, else Baostock.
  5. Regions = divergent cells (panel vs reference >10%) plus panel-NaN
     cells the reference covers (garbage eras are riddled with them). Each
     region is resolved as a whole:
       a. anchor = nearest both-valid panel/reference cell outside the
          region (real suspensions skipped) — ties levels across the splice
       b. panel values in the region agree with Baostock -> keep the panel,
          only fill its NaN holes from the reference
       c. else the reference agrees with Baostock -> adopt the reference
       d. else adopt Baostock itself
       e. adopted values are rescaled so the source equals the panel level
          at the anchor (splice continuous in the panel's qfq chain)
       f. no anchor / no judge coverage -> keep panel, NaN stays (safe)

     Failed earlier versions: naive cell-wise replacement imported Sina
     garbage (000630/000063, rolled back); anchor-only adoption adopted
     Sina era-wide garbage (000063 2020 corrupted to 2.4 vs true 35).

Run:  python panel_reconcile.py          (rebuilds + saves csi300_panel.pkl)
"""
import pickle
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest.loaders.registry import resolve_loader  # noqa: E402
import panel_scrub  # noqa: E402

DATA = Path(__file__).resolve().parent
PANEL = DATA / "csi300_panel.pkl"
BACKUP = Path("/tmp/csi300_panel.pkl.bak-20260814")
FIELDS = ["open", "high", "low", "close"]
DIV_BAND = (0.9, 1.1)        # panel/reference agreement band
ANCHOR_BAND = 0.20           # boundary anchor tolerance
ANCHOR_LOOK = 120            # max bars to search for an anchor/tie point
JUDGE_BAND = (0.85, 1.15)    # agreement band with the baostock judge
MIN_JUDGE_DAYS = 10          # minimum baostock coverage inside a region
VALIDATE_DAYS, MIN_VALIDATE = 250, 100


def prefixed(code: str) -> str:
    sym, suf = code.split(".")
    return ("sh" if suf == "SH" else "sz") + sym


def bs_symbol(code: str) -> str:
    sym, suf = code.split(".")
    return ("sh." if suf == "SH" else "sz.") + sym


def scrub_series(s: pd.Series) -> pd.Series:
    """panel_scrub's detector on a single series; returns cleaned series."""
    df = s.to_frame("c")
    p = {"close": df, "open": df.copy(), "high": df.copy(), "low": df.copy()}
    panel_scrub.scrub(p, verbose=False)
    return p["close"]["c"]


def runs_of(mask: pd.Series) -> list:
    """Maximal contiguous True runs as (start_pos, end_pos) inclusive."""
    pos = np.flatnonzero(mask.to_numpy())
    if len(pos) == 0:
        return []
    chunks = np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1)
    return [(int(c[0]), int(c[-1])) for c in chunks]


def fetch_baostock(bs, code: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_symbol(code), "date,open,high,low,close,volume",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def med_ratio(x: pd.Series, y: pd.Series, min_n: int):
    j = x.index.intersection(y.index)
    if len(j) < min_n:
        return None
    return float((x.loc[j] / y.loc[j]).median())


def resolve_region(a: int, b: int, p_close: pd.Series, ref: pd.Series,
                   bs_c: pd.Series, days: pd.DatetimeIndex):
    """Returns (source, scale): source in {'panel','ref','bs',None}.
    scale multiplies the source's values to tie them to the panel chain."""
    # anchor: nearest both-valid cell outside the region. Search the AFTER
    # side first, then the BEFORE side independently (a single chained loop
    # with break stops at the first valid cell even when it fails the band,
    # never reaching the other side — that dropped 600066.SH's 1171d region).
    anchor_pos = None
    for side in (range(b + 1, min(b + 1 + ANCHOR_LOOK, len(days))),
                 range(a - 1, max(a - 1 - ANCHOR_LOOK, -1), -1)):
        for nb in side:
            pv, rv = p_close.iloc[nb], ref.iloc[nb]
            if pd.notna(pv) and pd.notna(rv):
                if abs(pv / rv - 1) <= ANCHOR_BAND:
                    anchor_pos = nb
                break  # this side's nearest valid cell decided; try next side
        if anchor_pos is not None:
            break
    p_r = p_close.iloc[a:b + 1].dropna()
    ref_r = ref.iloc[a:b + 1].dropna()
    bs_r = bs_c.iloc[a:b + 1].dropna()

    src = None
    if len(p_r):  # b) panel agrees with the judge -> keep panel
        rp = med_ratio(p_r, bs_r, MIN_JUDGE_DAYS)
        if rp is not None and JUDGE_BAND[0] <= rp <= JUDGE_BAND[1]:
            src = "panel"
    if src is None:
        rr = med_ratio(ref_r, bs_r, MIN_JUDGE_DAYS)
        if len(ref_r) and rr is not None and JUDGE_BAND[0] <= rr <= JUDGE_BAND[1]:
            src = "ref"       # c) reference agrees with the judge
        elif len(bs_r) >= MIN_JUDGE_DAYS:
            src = "bs"        # d) the judge itself
        else:
            return None, None  # f) no judge coverage
    if src == "panel":
        return "panel", 1.0

    if anchor_pos is None:
        # No panel anchor (region touches the panel start or both boundaries
        # disagree). Two independent sources agreeing on the absolute level
        # (ref~bs) need no anchor; otherwise trust the judge's own level.
        return src, 1.0
    anchor_level = float(p_close.iloc[anchor_pos])

    # e) rescale the chosen source to the panel level at the anchor
    series = ref if src == "ref" else bs_c
    v = series.iloc[anchor_pos]
    if pd.notna(v) and v > 0:
        return src, anchor_level / float(v)
    # source missing at the anchor: tie at the nearest both-valid cell
    for nb in chain(range(b + 1, min(b + 1 + ANCHOR_LOOK, len(days))),
                    range(a - 1, max(a - 1 - ANCHOR_LOOK, -1), -1)):
        pv, sv = p_close.iloc[nb], series.iloc[nb]
        if pd.notna(pv) and pd.notna(sv) and sv > 0:
            return src, float(pv) / float(sv)
    return None, None


def reconcile_stock(code: str, p_close: pd.Series, sina: pd.DataFrame,
                    bs_df: pd.DataFrame, days: pd.DatetimeIndex) -> tuple:
    """Returns (replace: Series of new close values (NaN = no change), counts)."""
    replace = pd.Series(np.nan, index=days)
    counts = {"panel_holes_filled": 0, "ref": 0, "bs": 0}
    sc_raw = (sina["close"].reindex(days)
              if sina is not None and not sina.empty
              else pd.Series(np.nan, index=days))
    bs_c = (bs_df["close"].reindex(days) if not bs_df.empty
            else pd.Series(np.nan, index=days))

    sina_ok = False
    pcr, scr = p_close.iloc[-VALIDATE_DAYS:], sc_raw.iloc[-VALIDATE_DAYS:]
    valid = pcr.notna() & scr.notna()
    if int(valid.sum()) >= MIN_VALIDATE:
        ratios = pcr[valid] / scr[valid]
        med = float(ratios.median())
        sina_ok = (0.98 <= med <= 1.02 and
                   float(ratios.between(*DIV_BAND).mean()) >= 0.95)

    sc = scrub_series(sc_raw) if sina_ok else pd.Series(np.nan, index=days)
    ref = sc if sina_ok else bs_c
    both = p_close.notna() & ref.notna()
    diverge = both & ~(p_close / ref).between(*DIV_BAND)
    region = diverge | (p_close.isna() & ref.notna())

    for (a, b) in runs_of(region):
        src, scale = resolve_region(a, b, p_close, ref, bs_c, days)
        if src is None:
            continue
        if src == "panel":
            # keep panel; fill its NaN holes from the reference (rescaled)
            vals = ref.iloc[a:b + 1] * scale
            holes = p_close.iloc[a:b + 1].isna() & vals.notna()
            if holes.any():
                seg = replace.iloc[a:b + 1].copy()
                seg[holes] = vals[holes]
                replace.iloc[a:b + 1] = seg
                counts["panel_holes_filled"] += int(holes.sum())
        else:
            base = (ref if src == "ref" else bs_c).iloc[a:b + 1]
            replace.iloc[a:b + 1] = base * scale
            counts[src] += int(base.notna().sum())
    return replace, counts


def main() -> None:
    # Original flow started from the pristine pre-scrub backup; that /tmp file
    # does not survive reboots. Since build_csi300_panel.py now always scrubs,
    # the normal flow is: build (scrubbed panel) -> reconcile fills scrub-masked
    # holes and fixes residual splices, using the same judge arbitration.
    src = BACKUP if BACKUP.exists() else PANEL
    panel = pickle.load(open(src, "rb"))
    print(f"step 1: scrub panel (source: {src.name})", flush=True)
    scrub_report = panel_scrub.scrub(panel)
    close = panel["close"]
    days = close.index

    import baostock as bs
    import time as _time
    for _attempt in range(5):  # baostock has transient login failures (pitfall #18)
        if bs.login().error_code == "0":
            break
        _time.sleep(5)
    else:
        raise SystemExit("baostock login failed after 5 attempts")

    loader = resolve_loader("a_share")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    stats = {"stocks_changed": 0, "cells_changed": 0, "by_source": {},
             "notes": []}

    for n, code in enumerate(sorted(close.columns), 1):
        try:
            sina = loader._fetch_sina(prefixed(code), "2018-01-01", end)
        except Exception as exc:
            sina = None
            stats["notes"].append(f"{code}: sina fetch error {exc}")
        try:
            bs_df = fetch_baostock(bs, code, "2018-01-01", end)
        except Exception as exc:
            bs_df = pd.DataFrame()
            stats["notes"].append(f"{code}: baostock fetch error {exc}")
        if (sina is None or sina.empty) and bs_df.empty:
            continue
        replace, counts = reconcile_stock(code, close[code], sina, bs_df, days)
        take = days[replace.notna().to_numpy()]
        if len(take) == 0:
            continue
        stats["stocks_changed"] += 1
        stats["cells_changed"] += len(take)
        for k, v in counts.items():
            stats["by_source"][k] = stats["by_source"].get(k, 0) + v
        panel["close"].loc[take, code] = replace.loc[take]
        # OHLC follow the close's source where that source has them
        src_df = sina if (sina is not None and not sina.empty and
                          counts.get("ref", 0) >= counts.get("bs", 0)) else bs_df
        if src_df is not None and not src_df.empty:
            for f in ("open", "high", "low"):
                if f in src_df.columns:
                    src = src_df[f].reindex(days).loc[take]
                    good = src.notna()
                    panel[f].loc[take[good], code] = src[good]
        if n % 40 == 0:
            print(f"  {n}/{len(close.columns)} done", flush=True)

    bs.logout()
    panel["vwap"] = sum(panel[f] for f in FIELDS) / 4.0
    panel["_meta"] = {"universe": "csi300-ashare",
                      "constituent_count": int(close.shape[1]),
                      "price_adjustment": "qfq", "survivorship_bias": True,
                      "scrub": scrub_report,
                      "reconcile": {"stocks_changed": stats["stocks_changed"],
                                    "cells_changed": stats["cells_changed"],
                                    "by_source": stats["by_source"],
                                    "notes": stats["notes"][:50]}}
    pickle.dump(panel, open(PANEL, "wb"))
    print(f"\ncells changed: {stats['cells_changed']} across "
          f"{stats['stocks_changed']} stocks; by source: {stats['by_source']}")
    if stats["notes"]:
        print(f"notes ({len(stats['notes'])}): " + "; ".join(stats["notes"][:10]))
    print(f"SAVED {PANEL}")


if __name__ == "__main__":
    main()

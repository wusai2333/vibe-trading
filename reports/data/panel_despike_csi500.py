"""Post-reconcile despike: repair physically impossible one-day returns.

After build + reconcile, residual impossible returns remain (08-17: 610
cells, 557 of them 12-15% main-board moves). Their shape is not isolated
splice jumps but smooth multi-day FAKE RAMPS (e.g. 002532.SZ 2019-03: five
consecutive +13% days) — mis-adjusted Sina eras that reconcile adopted
because panel and Sina agreed with each other, so the Baostock judge was
never consulted. A real A-share cannot move beyond its daily limit on a
normal session (first-day-back from a LONG suspension excepted), so every
such cell is provably corrupt.

Fix per affected stock: merge flagged cells into runs (gap <= 2 days, edges
expanded by 1), fetch Baostock qfq once, rescale it to the panel's chain at
the nearest both-valid anchor outside the run (same anchor logic as
panel_reconcile), and overwrite the run. Cells Baostock cannot cover are
masked to NaN (safer than keeping provably false prices). Idempotent.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path(__file__).resolve().parent
PANEL = DATA / "csi500_panel.pkl"
FIELDS = ["open", "high", "low", "close"]
ANCHOR_LOOK = 120
ANCHOR_BAND = 0.20
MERGE_GAP, EDGE_PAD = 2, 1


def bs_symbol(code: str) -> str:
    sym, suf = code.split(".")
    return ("sh." if suf == "SH" else "sz.") + sym


def impossible_mask(close: pd.DataFrame) -> pd.DataFrame:
    days = close.index
    ret = close.pct_change()
    lim = pd.DataFrame(0.10, index=days, columns=close.columns)
    star = [c for c in close.columns if c.startswith("688")]
    gem = [c for c in close.columns if c.startswith("30")]
    if star:
        lim[star] = 0.20
    if gem:
        lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
    first_back = close.notna() & close.shift(1).isna()
    long_gap = first_back & close.shift(20).isna()
    return (ret.abs() > lim + 0.02) & ~long_gap


def runs_of(mask: pd.Series, merge_gap: int, pad: int, n: int):
    pos = np.flatnonzero(mask.to_numpy())
    if len(pos) == 0:
        return []
    chunks = np.split(pos, np.flatnonzero(np.diff(pos) > merge_gap) + 1)
    return [(max(0, int(c[0]) - pad), min(n - 1, int(c[-1]) + pad)) for c in chunks]


def fetch_baostock(bs, code: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_symbol(code), "date,open,high,low,close",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def main() -> None:
    panel = pickle.load(open(PANEL, "rb"))
    close = panel["close"]
    days = close.index
    imp = impossible_mask(close)
    before = int(imp.sum().sum())
    affected = imp.columns[imp.any()].tolist()
    print(f"before: {before} impossible cells across {len(affected)} stocks", flush=True)
    if not affected:
        print("nothing to do")
        return

    import baostock as bs
    import time as _time
    for _attempt in range(5):  # baostock has transient login failures (pitfall #18)
        if bs.login().error_code == "0":
            break
        _time.sleep(5)
    else:
        raise SystemExit("baostock login failed after 5 attempts")
    start, end = days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d")

    fixed_cells, masked_cells, no_cover = 0, 0, 0
    for n, code in enumerate(affected, 1):
        try:
            bs_df = fetch_baostock(bs, code, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  {code}: baostock error {exc}", file=sys.stderr)
            bs_df = pd.DataFrame()
        bs_c = bs_df["close"].reindex(days) if not bs_df.empty else pd.Series(np.nan, index=days)
        p_c = close[code]
        for (a, b) in runs_of(imp[code], MERGE_GAP, EDGE_PAD, len(days)):
            seg_bs = bs_c.iloc[a:b + 1]
            if int(seg_bs.notna().sum()) == 0:
                no_cover += (b - a + 1)
                panel["close"].iloc[a:b + 1, panel["close"].columns.get_loc(code)] = np.nan
                masked_cells += (b - a + 1)
                continue
            # anchor: nearest both-valid cell outside the run
            anchor, scale = None, None
            for side in (range(b + 1, min(b + 1 + ANCHOR_LOOK, len(days))),
                         range(a - 1, max(a - 1 - ANCHOR_LOOK, -1), -1)):
                for nb in side:
                    pv, bv = p_c.iloc[nb], bs_c.iloc[nb]
                    if pd.notna(pv) and pd.notna(bv) and bv > 0:
                        if abs(pv / bv - 1) <= ANCHOR_BAND:
                            anchor, scale = nb, float(pv) / float(bv)
                        break
                if anchor is not None:
                    break
            if scale is None:
                # no agreeing anchor: trust baostock's own level (its qfq
                # chain is the independent judge)
                scale = 1.0
            new_vals = seg_bs * scale
            take = days[a:b + 1][new_vals.notna().to_numpy()]
            if len(take):
                panel["close"].loc[take, code] = new_vals.loc[take]
                for f in ("open", "high", "low"):
                    if f in bs_df.columns:
                        src = bs_df[f].reindex(days).loc[take] * scale
                        panel[f].loc[take[src.notna()], code] = src[src.notna()]
                fixed_cells += int(new_vals.notna().sum())
            holes = new_vals.isna()
            if holes.any():
                panel["close"].loc[days[a:b + 1][holes.to_numpy()], code] = np.nan
                masked_cells += int(holes.sum())
        if n % 40 == 0:
            print(f"  {n}/{len(affected)} done", flush=True)

    bs.logout()
    panel["vwap"] = sum(panel[f] for f in FIELDS) / 4.0
    after = int(impossible_mask(panel["close"]).sum().sum())
    panel.setdefault("_meta", {})["despike"] = {
        "before": before, "after": after, "fixed_cells": fixed_cells,
        "masked_cells": masked_cells, "no_baostock_cover": no_cover,
        "date": pd.Timestamp.today().strftime("%Y-%m-%d")}
    pickle.dump(panel, open(PANEL, "wb"))
    print(f"fixed {fixed_cells} cells from baostock, masked {masked_cells} "
          f"(no cover: {no_cover}); impossible {before} -> {after}")
    print(f"SAVED {PANEL}")


if __name__ == "__main__":
    main()

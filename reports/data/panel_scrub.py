"""Detect and mask corrupted price segments in the CSI300 panel.

The dual-source (Sina+Tencent) qfq stitching occasionally splices in segments
whose price level is off by 10-50x (unadjusted or unit-mangled bars). Verified
examples: 601225.SH at 0.04-0.26 in 2019-20 (real ~8), 600039.SH at 0.21 in
2020-03 (real ~7), 000408.SZ at 0.43-1.0 in 2019-21 (real ~10-20), 600066.SH
at 0.67-1.9 in 2022 (real ~10-15). These produce impossible one-day returns
(+965%, +5200%) that poison backtests: the constrained backtest's deep-fill
once bought into one and booked a phantom +965% day.

Detection: a cell is corrupt when its close deviates too far from a centered
rolling median of the same stock, checked at three window scales (121/252/504
days). Short windows localize in dense regions; long windows still anchor in
sparse (heavily suspended) regions where short medians go NaN. Thresholds are
ASYMMETRIC: every observed corruption splices segments 10-50x too LOW, while
legitimate multi-month rallies can sit ~3x above a wide centered median
(verified false positives at a symmetric 3x: 300408.SZ / 002384.SZ June-2026
rally on the 504d window). So the low side is flagged at ratio < 1/3 on all
windows; the high side only at ratio > 3/4/5 for the 121/252/504 windows. A stock's
first 30 valid bars are exempt from the WIDE windows only (later levels make
IPO-window prices look low; verified 688126.SH first week) — the 121d window
still applies there and catches IPO-window garbage anchored on same-period
prices (verified 603259.SH 6.8 vs issue price 21.6). Runs iterate to a
fixpoint because long segments contaminate their own median.

Masked cells become NaN (same treatment as trading suspensions), which every
downstream tool already handles: factors drop the stock, the tradability mask
excludes it, and backtests earn 0 on it.

Usage:
    import panel_scrub; panel_scrub.scrub(panel)      # in build scripts
    python panel_scrub.py                              # clean cached pkl once
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent
PANEL = DATA / "csi300_panel.pkl"
FIELDS = ["open", "high", "low", "close"]
WINDOWS = (121, 252, 504)
LO, HI = 1 / 3, {121: 3.0, 252: 4.0, 504: 5.0}
IPO_EXEMPT_BARS = 30


def _ipo_exempt(close: pd.DataFrame) -> pd.DataFrame:
    """True for each stock's first N valid bars (IPO window)."""
    valid = close.notna().astype(int)
    cum = valid.cumsum()
    return valid.astype(bool) & (cum <= IPO_EXEMPT_BARS)


def bad_mask(close: pd.DataFrame, exempt: pd.DataFrame, minp: int = 20) -> pd.DataFrame:
    """The IPO exemption only shields the wide windows: a 121d median around an
    IPO anchors on the same early bars, so it still catches garbage there
    (verified: 603259.SH 6.8 vs issue 21.6) without harming legitimate debuts
    (688126.SH first week ratios ~0.9 at win=121)."""
    m = pd.DataFrame(False, index=close.index, columns=close.columns)
    for win in WINDOWS:
        med = close.rolling(win, center=True, min_periods=minp).median()
        ratio = close / med
        mw = (ratio < LO) | (ratio > HI[win])
        if win > 121:
            mw &= ~exempt
        m |= mw
    return m


def scrub(panel: dict, verbose: bool = True) -> dict:
    """Mask corrupt cells in-place; returns {'masked_cells': n, 'by_stock': {...}}."""
    close = panel["close"]
    exempt = _ipo_exempt(close)  # from the pristine data, fixed across passes
    total = 0
    combined = pd.DataFrame(False, index=close.index, columns=close.columns)
    while True:
        m = bad_mask(close.mask(combined), exempt)
        if not m.any().any():
            break
        combined |= m
        total = int(combined.sum().sum())
        if total > 5000:  # sanity valve: detection run amok
            raise RuntimeError(f"scrub would mask {total} cells; aborting")
    if total:
        for f in FIELDS:
            panel[f] = panel[f].mask(combined)
        if "vwap" in panel:
            panel["vwap"] = sum(panel[f] for f in FIELDS) / 4.0
    by_stock = combined.sum()
    report = {"masked_cells": total,
              "by_stock": {k: int(v) for k, v in by_stock[by_stock > 0].items()}}
    if verbose and total:
        print(f"scrub: masked {total} corrupt cells across "
              f"{len(report['by_stock'])} stocks: {report['by_stock']}")
    return report


if __name__ == "__main__":
    import sys
    # Rebuild from the pre-scrub backup when --from-backup is given, so the
    # improved detection runs on pristine data instead of layering masks.
    if "--from-backup" in sys.argv:
        bak = Path("/tmp/csi300_panel.pkl.bak-20260814")
        if not bak.exists():
            sys.exit("backup not found; run on the current panel instead")
        panel = pickle.load(open(bak, "rb"))
    else:
        panel = pickle.load(open(PANEL, "rb"))
    report = scrub(panel)
    panel.setdefault("_meta", {})["scrub"] = report
    pickle.dump(panel, open(PANEL, "wb"))
    print(f"SAVED {PANEL} ({report['masked_cells']} masked cells)")

"""One-off: rebuild holdings_history.csv rows for 2026-08-11/12/13.

The history file was recreated after the 08-13 report-deletion incident and
only contains 2026-08-14. Percentiles for the missing days are transcribed
from the published reports:
  - 08-11: reports/2026-08-12_持仓因子跟踪.md (comparison table)
  - 08-12: reports/2026-08-12_持仓因子跟踪.md (full table)
  - 08-13: reports/2026-08-13_稳定5因子推荐.md (holdings table)
Close prices / momenta for 08-11 and 08-13 are recomputed from the panel
with the same formulas holdings_tracker.py uses (close[-1]/close[-6]-1 etc.).
Scores for 08-11/08-13 are not recoverable (weights were date-specific) and
stay empty.
"""
import pickle
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
HISTORY = DATA / "holdings_history.csv"

panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
days = close.index

HOLDINGS = {
    "600011.SH": ("华能国际", 25),
    "601899.SH": ("紫金矿业", 20),
    "000657.SZ": ("中钨高新", 15),
    "600795.SH": ("国电电力", 10),
    "000408.SZ": ("藏格矿业", 8),
    "000975.SZ": ("中金黄金", 8),
    "601061.SH": ("中信金属", 7),
}

# pct_rank per date, transcribed from the reports cited above.
PCT = {
    "2026-08-11": {"600011.SH": 31.7, "601899.SH": 61.0, "000657.SZ": 83.8,
                   "600795.SH": 10.7, "000408.SZ": 93.4, "000975.SZ": 70.7,
                   "601061.SH": 85.9},
    "2026-08-13": {"600011.SH": 26.6, "601899.SH": 66.6, "000657.SZ": 85.5,
                   "600795.SH": 16.9, "000408.SZ": 82.1, "000975.SZ": 87.9,
                   "601061.SH": 67.6},
}

# 08-12 full row as published in 2026-08-12_持仓因子跟踪.md.
DAY_0812 = {
    "000408.SZ": (0.617, 93.1, -6.3, 13.5, 81.15),
    "000975.SZ": (0.565, 91.7, 9.3, 36.9, 26.48),
    "000657.SZ": (0.448, 84.1, 22.6, 0.9, 70.43),
    "601061.SH": (0.296, 74.1, -1.9, 12.3, 12.04),
    "601899.SH": (0.188, 68.6, -1.5, 16.5, 33.58),
    "600011.SH": (-0.086, 41.0, -2.4, 0.7, 6.95),
    "600795.SH": (-0.298, 22.1, 0.8, 6.3, 5.04),
}


def fin(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    import numpy as np
    return round(v, nd) if np.isfinite(v) else None


def rows_for(date_str):
    d = pd.Timestamp(date_str)
    assert d in days, f"{date_str} not a panel trading day"
    i = days.get_loc(d)
    out = []
    for sym, (name, wt) in HOLDINGS.items():
        c = close[sym] if sym in close.columns else None
        px = fin(c.iloc[i], 2) if c is not None else None
        r5 = fin((c.iloc[i] / c.iloc[i - 5] - 1) * 100, 1) if c is not None and i >= 5 else None
        r20 = fin((c.iloc[i] / c.iloc[i - 20] - 1) * 100, 1) if c is not None and i >= 20 else None
        out.append({"date": date_str, "symbol": sym, "name": name,
                    "weight_pct": wt, "score": None, "pct_rank": PCT[date_str][sym],
                    "mom_5d_pct": r5, "mom_20d_pct": r20, "close": px})
    return out


rebuilt = rows_for("2026-08-11") + rows_for("2026-08-13")
for sym, (name, wt) in HOLDINGS.items():
    score, pct, r5, r20, px = DAY_0812[sym]
    rebuilt.append({"date": "2026-08-12", "symbol": sym, "name": name,
                    "weight_pct": wt, "score": score, "pct_rank": pct,
                    "mom_5d_pct": r5, "mom_20d_pct": r20, "close": px})

df_old = pd.read_csv(HISTORY)
df_all = pd.concat([df_old, pd.DataFrame(rebuilt)], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["date", "symbol"], keep="last")
df_all = df_all.sort_values(["date", "pct_rank"], ascending=[True, False])
df_all.to_csv(HISTORY, index=False)
print(f"saved {len(df_all)} rows, dates: {sorted(df_all['date'].unique())}")

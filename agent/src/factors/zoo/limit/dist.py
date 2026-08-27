# ============================================================
# 中文名称: 距涨停距离（当日）
# 简要说明: 当日涨停幅度减去当日涨幅（归一化到涨停幅度），值越小越贴近涨停。
# 典型用途: 度量"收盘时的需求强度"：贴住涨停收盘 = 买盘强到收盘，
#          与盘中触板回落不同。收盘贴板股的次日行为是 A 股特色问题。
# ============================================================
"""limit DIST: normalized distance from today's close return to the limit-up.

``(lim - r_t) / lim`` in [0, ~2]: 0 means sealed at limit-up at the close,
larger values mean the close was further below the limit. Captures closing
demand intensity — a limit sealed into the close is a stronger signal than an
intraday touch that faded, and close-based formula alphas cannot tell them
apart (both days can share the same close-to-close return path history).
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'limit_dist',
    'nickname': 'normalized distance to limit-up at close',
    'theme': ['microstructure'],
    'formula_latex': r'(\mathrm{lim}_t - r_t) / \mathrm{lim}_t',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 2,
    'notes': '0 = sealed limit-up at close; board-aware limits.',
}


def _limit_pct(close: pd.DataFrame) -> pd.DataFrame:
    """Per-stock, per-date price-limit fraction (10% or 20%)."""
    lim = pd.DataFrame(0.10, index=close.index, columns=close.columns)
    star = [c for c in close.columns if c.startswith("688")]
    gem = [c for c in close.columns if c.startswith(("300", "301"))]
    if star:
        lim[star] = 0.20
    if gem:
        reform = pd.Timestamp("2020-08-24")
        lim.loc[close.index >= reform, gem] = 0.20
    return lim


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return normalized distance to limit-up on the supplied panel."""
    close = panel['close']
    ret = close.pct_change()
    lim = _limit_pct(close)
    return (lim - ret) / lim

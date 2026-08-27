# ============================================================
# 中文名称: 跌停次数（20日）
# 简要说明: 过去 20 个交易日触及跌停的天数（涨跌停幅度按板块规则）。
# 典型用途: 极端恐慌/抛压事件计数。可能是反转起点（恐慌出清），
#          也可能是下跌趋势确认。方向交给数据裁决。
# ============================================================
"""limit DNCNT20: count of limit-down days over the trailing 20 trading days."""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'limit_dncnt20',
    'nickname': '20-day limit-down day count',
    'theme': ['microstructure', 'reversal'],
    'formula_latex': r'\#\{t-19 \le i \le t : r_i \le -(\mathrm{lim}_i - 0.002)\}',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Board-aware A-share price limits; suspension days contribute 0.',
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
    """Return trailing-20d limit-down day count on the supplied panel."""
    close = panel['close']
    ret = close.pct_change()
    sealed_dn = ret <= -(_limit_pct(close) - 0.002)
    return sealed_dn.rolling(20, min_periods=12).sum()

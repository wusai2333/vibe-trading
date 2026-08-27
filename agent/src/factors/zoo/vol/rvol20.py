# ============================================================
# 中文名称: 实现波动率（20日）
# 简要说明: 过去 20 个交易日日收益的标准差。
# 典型用途: 低波异象（low-vol anomaly）基础因子：低波动股票未来收益反而更高，
#          在 A 股尤为显著（散户彩票偏好推高高波动股价格）。预期负 IC。
# ============================================================
"""vol RVOL20: 20-day realized volatility of daily returns.

Base factor for the low-volatility anomaly — low-vol stocks tend to earn
HIGHER subsequent returns, documented to be especially strong in China where
retail lottery demand overprices high-vol names. Expected negative IC; use
sign-aware (a reversed_strict verdict means it works with negative weight).
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'vol_rvol20',
    'nickname': '20-day realized volatility',
    'theme': ['volatility'],
    'formula_latex': r'\sigma\bigl(r_{t-19..t}\bigr)',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Low-vol anomaly base factor; negative IC expected.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing-20d return standard deviation on the supplied panel."""
    ret = panel['close'].pct_change()
    return ret.rolling(20, min_periods=12).std()

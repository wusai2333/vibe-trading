# ============================================================
# 中文名称: MAX彩票效应因子 (MAX20)
# 简要说明: 过去 20 日最大单日收益。彩票需求假说：极端正收益吸引散户追涨，高 MAX 股被高估、未来收益低。
# 典型用途: 预期负 IC（做空高 MAX）。Bali-Cakici-Whitelaw (2011)，中美均有实证。
# ============================================================
"""pool2 MAX20: maximum daily return over trailing 20 days.

Reference: Bali, Cakici, Whitelaw (2011), "Maxing out: Stocks as lotteries
and the cross-section of expected returns." Extreme recent up-days attract
retail lottery demand; high-MAX names tend to be overpriced and
underperform. Expected negative IC.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_max20',
    'nickname': '20-day MAX lottery effect',
    'theme': ['volatility'],
    'formula_latex': r'\max_{k\in[t-19,t]} r_k',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'Lottery-demand premium; expected negative IC.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 20-day maximum one-day return."""
    return panel['close'].pct_change().rolling(20, min_periods=15).max()

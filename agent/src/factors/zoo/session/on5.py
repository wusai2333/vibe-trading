# ============================================================
# 中文名称: 隔夜收益动量（5日）
# 简要说明: 过去 5 个交易日隔夜收益（开盘/昨收 - 1）的累计和。
# 典型用途: A 股隔夜收益承载情绪/信息动量，与日内收益的信息维度不同，
#          是价量公式因子（alpha101/gtja191/qlib158）未显式表达的方向。
# ============================================================
"""session ON5: cumulative 5-day overnight return (open / prev close - 1).

Decomposes each day's return into an overnight leg (close-to-open, driven by
news/sentiment arriving while the market is shut) and an intraday leg. This
factor stacks the overnight leg only, over 5 trading days. Orthogonal by
construction to close-to-close formula alphas: two stocks with identical
close paths can have very different overnight/intraday splits.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'session_on5',
    'nickname': '5-day cumulative overnight return',
    'theme': ['microstructure', 'momentum'],
    'formula_latex': r'\sum_{i=0}^{4}\bigl(\mathrm{open}_{t-i} / \mathrm{close}_{t-i-1} - 1\bigr)',
    'columns_required': ['open', 'close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 6,
    'notes': (
        'Overnight leg of daily returns summed over 5 trading days. NaN when '
        'the previous close is unavailable (suspension gap), so resumption '
        'days do not inject phantom gaps.'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return cumulative 5-day overnight return on the supplied OHLC panel."""
    overnight = panel['open'] / panel['close'].shift(1) - 1
    return overnight.rolling(5, min_periods=3).sum()

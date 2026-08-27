# ============================================================
# 中文名称: 隔夜-日内背离（20日）
# 简要说明: 20 日隔夜累计收益减去 20 日日内累计收益。
# 典型用途: 度量"盘外强/盘内弱"的背离：隔夜被资金追捧但日内持续被砸的股票，
#          与两条腿都强的股票行为不同。是两条腿信息的交互项。
# ============================================================
"""session ONIN20: 20-day overnight-minus-intraday divergence.

Interaction of the two session legs: positive when the stock is bid up
overnight (news/sentiment) but sold intraday (distribution), or vice versa.
Captures information neither leg carries alone.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'session_onin20',
    'nickname': '20-day overnight minus intraday return',
    'theme': ['microstructure'],
    'formula_latex': (
        r'\sum_{i=0}^{19}\bigl(\mathrm{open}_{t-i} / \mathrm{close}_{t-i-1} - 1\bigr)'
        r' - \sum_{i=0}^{19}\bigl(\mathrm{close}_{t-i} / \mathrm{open}_{t-i} - 1\bigr)'
    ),
    'columns_required': ['open', 'close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Overnight leg minus intraday leg over 20 trading days.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 20-day overnight-minus-intraday divergence on the OHLC panel."""
    overnight = panel['open'] / panel['close'].shift(1) - 1
    intraday = panel['close'] / panel['open'] - 1
    on = overnight.rolling(20, min_periods=12).sum()
    ind = intraday.rolling(20, min_periods=12).sum()
    return on - ind

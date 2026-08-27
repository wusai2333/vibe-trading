# ============================================================
# 中文名称: 隔夜收益动量（20日）
# 简要说明: 过去 20 个交易日隔夜收益（开盘/昨收 - 1）的累计和。
# 典型用途: 月度窗口的隔夜情绪动量，与 5 日版本互补（短/中两个尺度）。
# ============================================================
"""session ON20: cumulative 20-day overnight return (open / prev close - 1)."""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'session_on20',
    'nickname': '20-day cumulative overnight return',
    'theme': ['microstructure', 'momentum'],
    'formula_latex': r'\sum_{i=0}^{19}\bigl(\mathrm{open}_{t-i} / \mathrm{close}_{t-i-1} - 1\bigr)',
    'columns_required': ['open', 'close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Monthly-window overnight leg stack; see session_on5 for rationale.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return cumulative 20-day overnight return on the supplied OHLC panel."""
    overnight = panel['open'] / panel['close'].shift(1) - 1
    return overnight.rolling(20, min_periods=12).sum()

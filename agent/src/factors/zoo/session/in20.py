# ============================================================
# 中文名称: 日内收益累计（20日）
# 简要说明: 过去 20 个交易日日内收益（收盘/开盘 - 1）的累计和。
# 典型用途: A 股日内段倾向反转（散户驱动的盘中追涨杀跌），预期 IC 为负，
#          若 strict 判 reversed 则按负权重使用（与 qlib158_klow 同理）。
# ============================================================
"""session IN20: cumulative 20-day intraday return (close / open - 1).

The intraday leg is the retail-flow-dominated part of the A-share session;
documented tendency is mean reversion, so this factor is expected to earn a
NEGATIVE IC and would be used with a negative blend weight if it survives
the strict random control.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'session_in20',
    'nickname': '20-day cumulative intraday return',
    'theme': ['microstructure', 'reversal'],
    'formula_latex': r'\sum_{i=0}^{19}\bigl(\mathrm{close}_{t-i} / \mathrm{open}_{t-i} - 1\bigr)',
    'columns_required': ['open', 'close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Intraday leg stack; negative IC expected (reversal), use sign-aware.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return cumulative 20-day intraday return on the supplied OHLC panel."""
    intraday = panel['close'] / panel['open'] - 1
    return intraday.rolling(20, min_periods=12).sum()

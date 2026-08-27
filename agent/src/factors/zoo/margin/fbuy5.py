# ============================================================
# 中文名称: 融资买入强度 (FBUY5)
# 简要说明: 5 日均融资买入额 / 融资余额：杠杆仓位的换手强度，区分「加仓活跃」与「持仓躺平」。
# 典型用途: 杠杆交易活跃度。PIT 右移一日。
# ============================================================
"""margin fbuy5: 5d margin buying intensity.

PIT: margin data for day d is published after the d close, so every
    series is shifted by one trading day — the value used at time t is the
    margin state of t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'margin_fbuy5',
    'nickname': '5d margin buying intensity',
    'theme': ['sentiment'],
    'formula_latex': r'\mathrm{mean}_5(V^{fin}_{t-1})/B^{fin}_{t-1}',
    'columns_required': ['margin:fin_balance', 'margin:fin_buy'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 8,
    'notes': 'PIT shift(1) inside; turnover of leveraged book.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return PIT-safe 5-day margin buying intensity."""
    bal = panel['margin:fin_balance'].shift(1)
    buy = panel['margin:fin_buy'].shift(1)
    out = buy.rolling(5, min_periods=4).mean() / bal
    return out.replace([np.inf, -np.inf], np.nan)

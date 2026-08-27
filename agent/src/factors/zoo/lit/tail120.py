# ============================================================
# 中文名称: 尾部风险暴露 (TAIL120)
# 简要说明: 市场极端下跌日（滚动 120 日最低 5% 分位）个股的平均收益。
# 典型用途: 尾部风险暴露。Kelly-Jiang (2012) 族的日频简化。
# ============================================================
"""lit tail120: mean stock return on market tail days (bottom 5%), 120d.

Source: Kelly & Jiang (2012) tail-risk family, daily simplified version.
Exposure to the market's worst days within a rolling window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_tail120',
    'nickname': '120d market-tail-day exposure',
    'theme': ['volatility'],
    'formula_latex': r'E[r_i \mid r_m \in q_{0.05}^{120}]',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 120,
    'notes': 'Tail days: r_m at/below rolling 5th percentile.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return mean return on market tail days, rolling 120d."""
    ret = panel['close'].pct_change()
    rm = ret.mean(axis=1)
    q = rm.rolling(120, min_periods=80).quantile(0.05)
    mask = rm <= q
    out = ret.where(mask, np.nan).rolling(120, min_periods=5).mean()
    return out.replace([np.inf, -np.inf], np.nan)

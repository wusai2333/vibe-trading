# ============================================================
# 中文名称: 残差动量 (RESMOM20)
# 简要说明: 20 日动量剥离市场暴露：mom20 - beta120 × 市场 mom20。
# 典型用途: 纯个股动量。Blitz (2011) 残差动量的简化实现。
# ============================================================
"""lit resmom20: 20-day momentum net of market exposure.

Source: Blitz et al. (2011) residual momentum, simplified: subtract the
120-day beta times the market's own 20-day momentum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_resmom20',
    'nickname': '20d residual momentum',
    'theme': ['momentum'],
    'formula_latex': r'mom^{20}_i - \beta^{120}_i mom^{20}_m',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 140,
    'notes': 'Beta from 120d market regression (concurrent, rolling).',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return market-adjusted 20-day momentum."""
    ret = panel['close'].pct_change()
    rm = ret.mean(axis=1)
    beta = ret.rolling(120, min_periods=80).cov(rm).div(rm.rolling(120, min_periods=80).var(), axis=0)
    mom = panel['close'].pct_change(20)
    momm = rm.rolling(20).sum()
    out = mom.sub(beta.mul(momm, axis=0), axis=0)
    return out.replace([np.inf, -np.inf], np.nan)

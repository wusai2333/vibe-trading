# ============================================================
# 中文名称: 协偏度 (COSKEW60)
# 简要说明: 个股收益与市场收益平方的滚动协变：个股在市场大涨/大跌日的不对称暴露。
# 典型用途: 系统性偏度风险溢价。Harvey-Siddique (2000)。
# ============================================================
"""lit coskew60: 60-day coskewness with the equal-weight market.

Source: Harvey & Siddique (2000), conditional skewness in asset pricing.
Coskewness = E[(r_i - mu_i)(r_m - mu_m)^2] / (sigma_i sigma_m^2), rolling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_coskew60',
    'nickname': '60d coskewness vs market',
    'theme': ['volatility'],
    'formula_latex': r'\frac{E[(r_i-\mu_i)(r_m-\mu_m)^2]}{\sigma_i\sigma_m^2}',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Market = equal-weight panel mean return.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return rolling 60-day coskewness against the market."""
    ret = panel['close'].pct_change()
    rm = ret.mean(axis=1)
    dm = rm - rm.rolling(60, min_periods=40).mean()
    di = ret.sub(ret.rolling(60, min_periods=40).mean(), axis=0)
    num = di.mul(dm ** 2, axis=0).rolling(60, min_periods=40).mean()
    si = ret.rolling(60, min_periods=40).std()
    sm2 = rm.rolling(60, min_periods=40).std() ** 2
    out = num.div(si.mul(sm2, axis=0), axis=0)
    return out.replace([np.inf, -np.inf], np.nan)

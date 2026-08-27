# ============================================================
# 中文名称: 平均振幅因子 (RNG20)
# 简要说明: 过去 20 日 (high-low)/close 的均值：Parkinson 风格的区间振幅度量。
# 典型用途: 振幅/波动代理，与收益率波动率不同的信息维度（日内震荡强度）。
# ============================================================
"""pool2 RNG20: mean daily range over trailing 20 days.

Average of (high-low)/close, a Parkinson-style range measure of daily
amplitude. Captures intraday churn intensity distinct from close-to-close
return volatility.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_rng20',
    'nickname': '20d mean daily range',
    'theme': ['volatility'],
    'formula_latex': r'\mathrm{mean}_{20}\bigl((h_t-l_t)/c_t\bigr)',
    'columns_required': ['high', 'low', 'close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 20,
    'notes': 'Range amplitude; sign decided by bench.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 20-day mean normalized range."""
    rng = (panel['high'] - panel['low']) / panel['close']
    return rng.rolling(20, min_periods=15).mean()

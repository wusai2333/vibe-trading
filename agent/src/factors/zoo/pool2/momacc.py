# ============================================================
# 中文名称: 动量加速度因子 (MOMACC)
# 简要说明: 20日动量减去其 20 日前的值：动量在加速还是衰竭。
# 典型用途: 区分趋势延续与趋势末端；文献上动量变化对后续收益有增量信息。
# ============================================================
"""pool2 MOMACC: momentum acceleration (change of 20d momentum).

Second-difference momentum: rising momentum (acceleration) vs fading
momentum. Momentum-change information is not captured by level momentum
alone; related to momentum continuation/reversal literature.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_momacc',
    'nickname': 'momentum acceleration (mom20 - mom20 lag20)',
    'theme': ['momentum'],
    'formula_latex': r'(c_t/c_{t-20}-1) - (c_{t-20}/c_{t-40}-1)',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 10,
    'min_warmup_bars': 41,
    'notes': 'Change in 20d momentum; sign decided by bench.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 20d momentum minus its value 20 days ago."""
    mom = panel['close'].pct_change(20)
    return mom - mom.shift(20)

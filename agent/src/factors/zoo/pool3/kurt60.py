# ============================================================
# 中文名称: 收益分布峰度 (KURT60)
# 简要说明: 60 日收益峰度：肥尾/彩票属性度量，MAX 因子的分布族扩展。
# 典型用途: 尾部风险暴露。预期负向（肥尾彩票股跑输），检验定。
# ============================================================
"""pool3 kurt60: trailing 60-day return kurtosis.

Source: lottery-risk family extension (Bali-Cakici-Whitelaw MAX line;
extreme-view multifactor research). Fat-tail measure over the whole
distribution rather than the single max day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_kurt60',
    'nickname': '60d return kurtosis',
    'theme': ['volatility'],
    'formula_latex': r'\mathrm{kurt}_{60}(r_t)',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Fat-tail / lottery exposure; expected negative.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 60-day return kurtosis."""
    return panel['close'].pct_change().rolling(60, min_periods=40).kurt()

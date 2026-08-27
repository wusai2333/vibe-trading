# ============================================================
# 中文名称: 量价相关性因子 (VPCORR20)
# 简要说明: 过去 20 日收盘价与成交量的滚动相关系数：放量上涨/缩量下跌为正，量价背离为负。
# 典型用途: 度量量价确认。A 股量价关系有独立于动量的信息（08-19 预注册候选）。
# ============================================================
"""pool2 VPCORR20: 20-day rolling correlation of close and volume.

Positive when volume confirms price moves (rises on up-days), negative on
volume-price divergence. A volume-confirmation signal orthogonal in
construction to pure price factors.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_vpcorr20',
    'nickname': '20d close-volume correlation',
    'theme': ['volume'],
    'formula_latex': r'\mathrm{corr}_{20}(c_t, V_t)',
    'columns_required': ['close', 'volume'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 20,
    'notes': 'Rolling Pearson corr of close level and volume.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 20-day close-volume correlation."""
    return panel['close'].rolling(20, min_periods=15).corr(panel['volume'])

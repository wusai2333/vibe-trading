# ============================================================
# 中文名称: 下行波动率因子 (DNVOL60)
# 简要说明: 60 日内负收益的标准差（只算下跌日的波动）。
# 典型用途: 下行风险度量，Ang-Hodgkin-Kinney-Bali (2006) 系；预期负 IC。与 ivol60 互补（不剥 beta，只看下尾）。
# ============================================================
"""pool2 DNVOL60: 60-day downside volatility.

Standard deviation of floored (negative-only) daily returns over 60 days.
Downside risk measures carry incremental information beyond symmetric
volatility (Ang, Hodgkin, Kinney, Bali 2006 family). Expected negative IC.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_dnvol60',
    'nickname': '60d downside volatility',
    'theme': ['volatility'],
    'formula_latex': r'\sigma\bigl(\min(r_t,0)\bigr),\ 60d',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Std of min(ret,0); expected negative IC.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 60-day downside volatility."""
    ret = panel['close'].pct_change()
    return ret.clip(upper=0.0).rolling(60, min_periods=40).std()

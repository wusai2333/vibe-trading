# ============================================================
# 中文名称: 符号量能流因子 (OBV20)
# 简要说明: 过去 20 日 sign(收益)×成交量 的均值，除以同期均量：归一化的 OBV 资金流。
# 典型用途: 资金流方向度量（Chaikin/OBV 家族），量纲无关、横截面可比。
# ============================================================
"""pool2 OBV20: normalized signed volume flow over 20 days.

Mean of sign(return)*volume divided by mean volume, a scale-free OBV-style
money-flow direction measure. Cross-sectionally comparable across names of
different size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_obv20',
    'nickname': '20d signed volume flow',
    'theme': ['volume'],
    'formula_latex': r'\mathrm{mean}_{20}(\mathrm{sgn}(r_t)V_t)/\mathrm{mean}_{20}(V_t)',
    'columns_required': ['close', 'volume'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'Scale-free OBV slope proxy.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 20-day normalized signed volume flow."""
    ret = panel['close'].pct_change()
    vol = panel['volume']
    signed = np.sign(ret) * vol
    out = signed.rolling(20, min_periods=15).mean() / vol.rolling(20, min_periods=15).mean()
    return out.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 中文名称: 趋势一致性 (TREND)
# 简要说明: 5/10/20/60 日动量符号的一致性：多周期趋势共振。
# 典型用途: 趋势因子。Liu-Zhou (清华) Trend Factor in China 的简化。
# ============================================================
"""lit trend: multi-horizon momentum sign agreement.

Source: Liu & Zhou (Tsinghua), Trend Factor in China, simplified to the
mean sign of 5/10/20/60-day returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_trend',
    'nickname': 'multi-horizon trend agreement',
    'theme': ['momentum'],
    'formula_latex': r'\frac{1}{4}\sum_{k}\mathrm{sgn}(mom_k)',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Sign agreement over 5/10/20/60d horizons.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return mean sign of multi-horizon momentum."""
    close = panel['close']
    s = None
    for k in (5, 10, 20, 60):
        sg = np.sign(close.pct_change(k))
        s = sg if s is None else s + sg
    return s / 4

# ============================================================
# 中文名称: 下行贝塔 (DNBETA120)
# 简要说明: 仅用市场下跌日估计的 beta：熊市暴露。
# 典型用途: 下行风险溢价。Ang-Hodgkin-Kinney-Bali (2006)。
# ============================================================
"""lit dnbeta120: beta estimated on down-market days only, 120d window.

Source: Ang, Hodgkin, Kinney, Bali (2006), downside beta. Market =
equal-weight panel mean; down days = negative market return days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_dnbeta120',
    'nickname': '120d down-market beta',
    'theme': ['volatility'],
    'formula_latex': r'\beta_i^{down} = \frac{cov_{down}(r_i,r_m)}{var_{down}(r_m)}',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 120,
    'notes': 'Down days: r_m < 0; rolling 120d with min 50 down days.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return rolling down-market beta."""
    ret = panel['close'].pct_change()
    rm = ret.mean(axis=1)
    mask = rm < 0
    ri = ret.where(mask, np.nan)
    rmd = rm.where(mask, np.nan)
    cov = ri.rolling(120, min_periods=50).cov(rmd)
    var = rmd.rolling(120, min_periods=50).var()
    out = cov.div(var, axis=0)
    return out.replace([np.inf, -np.inf], np.nan)

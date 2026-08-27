# ============================================================
# 中文名称: 5日反转因子 (REV5)
# 简要说明: 过去 5 日收益取负。A 股短周期反转是全球文献中最强的日频异象之一（本项目 08-19 实测 raw IC -0.022, t -4.7）。
# 典型用途: 做多近一周输家。与 academic_strev（21日）不同周期，预期独立。
# ============================================================
"""pool2 REV5: 5-day return reversal.

Reference: Jegadeesh (1990); short-horizon reversal is among the strongest
daily anomalies documented on China A-shares. Distinct horizon from
academic_strev (21-day). Expected negative IC on raw momentum, hence the
sign flip so that higher values = larger recent losers.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_rev5',
    'nickname': '5-day reversal',
    'theme': ['reversal'],
    'formula_latex': r'\mathrm{zscore}_{x}\bigl(-(c_t/c_{t-5}-1)\bigr)',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 6,
    'notes': 'Raw 5d return sign-flipped; bench decides sign.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return negative trailing 5-day return."""
    return -(panel['close'].pct_change(5))

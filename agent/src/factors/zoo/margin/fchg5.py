# ============================================================
# 中文名称: 融资余额5日增速 (FCHG5)
# 简要说明: 融资余额的 5 日增长率，度量杠杆资金短期流入加速度。海通因子研究 7：融资增速比融资余额水平更有选股信息。
# 典型用途: 杠杆资金动量。PIT 右移一日。预期方向由检验定。
# ============================================================
"""margin fchg5: 5d margin-financing balance growth.

PIT: margin data for day d is published after the d close, so every
    series is shifted by one trading day — the value used at time t is the
    margin state of t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'margin_fchg5',
    'nickname': '5d margin-financing balance growth',
    'theme': ['sentiment'],
    'formula_latex': r'\frac{B^{fin}_{t-1}}{B^{fin}_{t-6}}-1',
    'columns_required': ['margin:fin_balance'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 8,
    'notes': 'PIT shift(1) inside; Haitong: growth beats level.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return PIT-safe 5-day growth of margin financing balance."""
    bal = panel['margin:fin_balance'].shift(1)
    out = bal.pct_change(5)
    return out.replace([np.inf, -np.inf], np.nan)

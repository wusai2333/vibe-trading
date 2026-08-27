# ============================================================
# 中文名称: 净利润断层 (JOR)
# 简要说明: 公告日跳空 × 预告方向，20 日线性衰减。天风金工净利润断层因子的实现。
# 典型用途: 基本面与技术面共振的事件动量。
# ============================================================
"""event jor: jump-on-result (announcement-day gap x surprise sign, 20d decay).

Source: Tianfeng Securities net-profit-gap (JOR) research; the gap on the
announcement day signed by the preview direction, decaying linearly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'event_jor',
    'nickname': 'jump-on-result earnings gap',
    'theme': ['growth'],
    'formula_latex': r'\mathrm{gap}_{ann}\cdot\mathrm{sgn}(\mathrm{surprise})',
    'columns_required': ['event:jor'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': 'Pre-built with 20-day linear decay in the event panel.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the decaying jump-on-result series."""
    return panel['event:jor']

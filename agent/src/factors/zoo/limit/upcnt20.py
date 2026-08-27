# ============================================================
# 中文名称: 涨停次数（20日）
# 简要说明: 过去 20 个交易日触及涨停的天数（主板 10%、创/科 20%，贴住 ≤0.2pp 计）。
# 典型用途: A 股本土微观结构因子：涨停是极端需求事件，既可能是动量延续
#          （资金接力），也可能是过热见顶信号。方向交给数据裁决。
# ============================================================
"""limit UPCNT20: count of limit-up days over the trailing 20 trading days.

A-share native microstructure factor. Price limits are binding constraints unique
to this market; a sealed limit-up is an extreme demand event that close-based
formula alphas (alpha101/gtja191/qlib158) never observe directly. Limits:
main board 10%; ChiNext (300/301) 20% since the 2020-08-24 registration
reform (10% before); STAR (688) always 20%. A day counts as limit-up when
its return is within 0.2pp of the board limit.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'limit_upcnt20',
    'nickname': '20-day limit-up day count',
    'theme': ['microstructure', 'momentum'],
    'formula_latex': r'\#\{t-19 \le i \le t : r_i \ge \mathrm{lim}_i - 0.002\}',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 21,
    'notes': 'Board-aware A-share price limits; suspension days contribute 0.',
}


def _limit_pct(close: pd.DataFrame) -> pd.DataFrame:
    """Per-stock, per-date price-limit fraction (10% or 20%)."""
    lim = pd.DataFrame(0.10, index=close.index, columns=close.columns)
    star = [c for c in close.columns if c.startswith("688")]
    gem = [c for c in close.columns if c.startswith(("300", "301"))]
    if star:
        lim[star] = 0.20
    if gem:
        reform = pd.Timestamp("2020-08-24")
        lim.loc[close.index >= reform, gem] = 0.20
    return lim


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing-20d limit-up day count on the supplied panel."""
    close = panel['close']
    ret = close.pct_change()
    sealed_up = ret >= _limit_pct(close) - 0.002
    return sealed_up.rolling(20, min_periods=12).sum()

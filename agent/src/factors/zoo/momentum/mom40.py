# 61-21 skip-window momentum (40d signal), archived 2026-08-21 from the CST
# recipe tests. Their finding: A-share momentum lives at ~40d windows, not the
# US-style 12-1m. On our CSI300 daily yardstick it is dead (IC 0.0016) — kept
# for the record per the lit-zoo precedent; any revival needs a new horizon.
"""momentum_mom40: formula = \\mathrm{close}_{t-21} / \\mathrm{close}_{t-61} - 1."""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'momentum_mom40',
    'nickname': '61-21 skip momentum (40d window, CST recipe)',
    'theme': ['momentum'],
    'formula_latex': '\\mathrm{close}_{t-21} / \\mathrm{close}_{t-61} - 1',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 21,
    'min_warmup_bars': 61,
    'notes': (
        'Skip-window momentum: 40-day signal ending 21 bars ago (avoids '
        'short-term reversal). Dead on CSI300 daily yardstick (IC 0.0016, '
        '2026-08-21 CST practice test); archived for the record.'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 61-21 skip momentum on the supplied panel."""
    close = panel['close']
    return close.shift(21) / close.shift(61) - 1.0

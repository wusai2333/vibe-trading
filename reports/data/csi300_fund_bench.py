"""Strict bench for the fundamental zoo on the baostock PIT panel.

Loads the price panel + fund_cache.pkl (built by build_fund_panel.py),
injects the fund:* columns, and runs bench_runner_strict (random control +
OOS split) over the fundamental zoo: fund_earnings_yield, fund_roe,
fund_gross_profitability, fund_asset_growth, fund_bp.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for k, v in fund.items():
    if k.startswith("fund:"):
        panel[k] = v
print(f"panel enriched: {sum(1 for k in panel if k.startswith('fund:'))} fund columns",
      file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner_strict import run_bench_strict
t0 = time.time()
r = run_bench_strict(zoo="fundamental", universe="csi300-ashare",
                     period="2018-01-01/2025-12-31",
                     random_control=True, n_random_seeds=5,
                     oos_split="2022-06-30", top=20)
print(f"tested={r.get('n_alphas_tested')} confirmed_alive={r.get('confirmed_alive')} "
      f"wall={time.time()-t0:.0f}s")
rows = {}
for row in r.get("rows", []):
    if isinstance(row, dict):
        rows[row["id"]] = row
        print(json.dumps({k: row.get(k) for k in
            ("id", "_category", "ir", "ic_mean", "random_ic_mean", "oos_ir",
             "ic_positive_ratio")}, ensure_ascii=False))
json.dump(rows, open(DATA / "csi300_fund_strict.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print(f"SAVED {DATA / 'csi300_fund_strict.json'}", file=sys.stderr)

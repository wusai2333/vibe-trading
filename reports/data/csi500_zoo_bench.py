"""Strict bench for limit/vol/session zoos on the CSI500 panel.

CSI500 pilot step 2 (after fundamentals closed): do the price-based
microstructure/vol/session families behave differently on mid-caps?
Same protocol as the CSI300 runs: random control (5 seeds) + OOS split.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi500_panel.pkl", "rb"))
print(f"panel: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner_strict import run_bench_strict
rows = {}
for zoo in ("limit", "vol", "session"):
    t0 = time.time()
    r = run_bench_strict(zoo=zoo, universe="csi500-ashare",
                         period="2018-01-01/2025-12-31",
                         random_control=True, n_random_seeds=5,
                         oos_split="2022-06-30", top=20)
    print(f"[{zoo}] tested={r.get('n_alphas_tested')} "
          f"confirmed_alive={r.get('confirmed_alive')} wall={time.time()-t0:.0f}s",
          file=sys.stderr)
    for row in r.get("rows", []):
        if isinstance(row, dict):
            rows[row["id"]] = row
            print(json.dumps({k: row.get(k) for k in
                ("id", "_category", "ir", "ic_mean", "random_ic_mean",
                 "alpha_t_train", "alpha_t_test", "ic_positive_ratio")},
                ensure_ascii=False))
json.dump(rows, open(DATA / "csi500_zoo_strict.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print(f"SAVED {DATA / 'csi500_zoo_strict.json'}", file=sys.stderr)

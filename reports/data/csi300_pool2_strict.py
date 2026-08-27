"""Strict bench for the pool2 expanded-factor zoo on the CLEAN CSI300 panel.

Pool-expansion round 1 (2026-08-19, user directed): 10 pre-registered
OHLCV factors from the literature (reversal/lottery/momentum-path/volume-
flow/downside-vol/range/vol-regime), none previously in the zoo. Protocol
identical to limit/vol/session runs: random control (5 seeds) + OOS split.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
print(f"panel: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner_strict import run_bench_strict
t0 = time.time()
r = run_bench_strict(zoo="pool2", universe="csi300-ashare",
                     period="2018-01-01/2026-08-19",
                     random_control=True, n_random_seeds=5,
                     oos_split="2022-06-30", top=20)
print(f"[pool2] tested={r.get('n_alphas_tested')} confirmed_alive={r.get('confirmed_alive')} "
      f"wall={time.time()-t0:.0f}s", file=sys.stderr)
rows = {}
for row in r.get("rows", []):
    if isinstance(row, dict):
        rows[row["id"]] = row
        print(json.dumps({k: row.get(k) for k in
            ("id", "_category", "ir", "ic_mean", "random_ic_mean",
             "alpha_t_train", "alpha_t_test", "ic_positive_ratio")},
            ensure_ascii=False, default=str))
json.dump(rows, open(DATA / "csi300_pool2_strict.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print(f"SAVED {DATA / 'csi300_pool2_strict.json'}", file=sys.stderr)
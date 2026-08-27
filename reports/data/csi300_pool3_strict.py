"""Strict bench for the pool3 zoo (modsearch-sourced candidates, 2026-08-20).

Same protocol as pool2/margin runs: random control (5 seeds) + OOS split.
Injects margin:* columns for pool3_mfinvol.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
margin = pickle.load(open(DATA / "margin_panel.pkl", "rb"))
for k, v in margin.items():
    panel[f"margin:{k}"] = v.reindex(panel["close"].index)
print(f"panel enriched: {sum(1 for k in panel if k.startswith('margin:'))} margin columns",
      file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner_strict import run_bench_strict
t0 = time.time()
r = run_bench_strict(zoo="pool3", universe="csi300-ashare",
                     period="2018-01-01/2026-08-20",
                     random_control=True, n_random_seeds=5,
                     oos_split="2022-06-30", top=20)
print(f"tested={r.get('n_alphas_tested')} confirmed_alive={r.get('confirmed_alive')} "
      f"wall={time.time()-t0:.0f}s", file=sys.stderr)
rows = {}
for row in r.get("rows", []):
    if isinstance(row, dict):
        rows[row["id"]] = row
        print(json.dumps({k: row.get(k) for k in
            ("id", "_category", "ir", "ic_mean", "random_ic_mean",
             "alpha_t_train", "alpha_t_test", "ic_positive_ratio")},
            ensure_ascii=False, default=str))
json.dump(rows, open(DATA / "csi300_pool3_strict.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print(f"SAVED {DATA / 'csi300_pool3_strict.json'}", file=sys.stderr)

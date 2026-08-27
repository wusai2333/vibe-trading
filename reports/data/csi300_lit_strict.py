"""Strict bench for the lit zoo (journal-family factors, 2026-08-20).

Final pre-registered round: coskew/dnbeta/tail/resmom/psliq/trend.
Same protocol: random control (5 seeds) + OOS split. Price-derived only.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner_strict import run_bench_strict
t0 = time.time()
r = run_bench_strict(zoo="lit", universe="csi300-ashare",
                     period="2018-01-01/2026-08-20",
                     random_control=True, n_random_seeds=5,
                     oos_split="2022-06-30", top=20)
print("tested=" + str(r.get('n_alphas_tested')) + " confirmed_alive=" + str(r.get('confirmed_alive'))
      + " wall=" + str(round(time.time()-t0)) + "s", file=sys.stderr)
rows = {}
for row in r.get("rows", []):
    if isinstance(row, dict):
        rows[row["id"]] = row
        print(json.dumps({k: row.get(k) for k in
            ("id", "_category", "ir", "ic_mean", "random_ic_mean",
             "alpha_t_train", "alpha_t_test", "ic_positive_ratio")},
            ensure_ascii=False, default=str))
json.dump(rows, open(DATA / "csi300_lit_strict.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("SAVED csi300_lit_strict.json", file=sys.stderr)

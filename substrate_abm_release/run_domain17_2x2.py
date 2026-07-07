"""
run_domain17_2x2.py
===================
Domain17 substrate 2x2, parameterized by axis (tree or markov). Same 8 hard
scenarios in both runs (from hard_scenarios_domain17.pkl); output goes to a
distinct file per axis. Checkpointed per scenario.

Usage:
    python run_domain17_2x2.py tree
    python run_domain17_2x2.py markov
"""
from __future__ import annotations
import json
import os
import pickle
import sys
import time
import numpy as np

from domain17 import make_domain17, make_observability17
from experiment import TeamConfig, run_many


IN_PATH = "./results/hard_scenarios_domain17.pkl"
THD = 0.55
N_RUNS = 40


def run_cell(spec, obs, scenario, engines, projection, n_runs, seed):
    tc = TeamConfig(engines=engines, projection_mode=projection,
                    charitable=False, thd=THD, gamma=0.95,
                    max_actions=100, idle_limit=5)
    return run_many(spec, obs, scenario, tc, n_runs=n_runs, seed=seed)


def run_2x2(spec, obs, scenario, axis, seed_offset, base_seed):
    """axis in {'tree', 'markov'} -> composition mismatched = 1 <axis> + 2 BBN."""
    cells = {
        "matched-aware":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="aware"),
        "matched-naive":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="naive"),
        "mismatched-aware": dict(engines={"A":axis,"B":"bbn","C":"bbn"},    projection="aware"),
        "mismatched-naive": dict(engines={"A":axis,"B":"bbn","C":"bbn"},    projection="naive"),
    }
    out = {}
    for i, (name, cfg) in enumerate(cells.items()):
        r = run_cell(spec, obs, scenario, cfg["engines"], cfg["projection"],
                     n_runs=N_RUNS, seed=base_seed + seed_offset + i)
        out[name] = {"final_mean": r["final_mean"], "final_sd": r["final_sd"],
                     "actions_mean": r["actions_mean"]}
    return out


def load_partial(out_path, axis):
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {"thd": THD, "n_runs": N_RUNS, "results": [], "axis": axis, "domain": "domain17"}


def save_partial(out_path, data):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("tree", "markov"):
        print("Usage: python run_domain17_2x2.py {tree|markov}")
        sys.exit(1)
    axis = sys.argv[1]
    out_path = f"./results/domain17_{axis}_results.json"
    base_seed = {"tree": 900000, "markov": 910000}[axis]

    spec = make_domain17()
    obs = make_observability17()
    with open(IN_PATH, "rb") as f:
        payload = pickle.load(f)
    scenarios = payload["scenarios"]
    print(f"domain17 {axis}-axis 2x2 on {len(scenarios)} hard scenarios, "
          f"THD={THD}, N={N_RUNS}\n")

    data = load_partial(out_path, axis)
    done = {r["rank"] for r in data["results"]}
    print(f"Already done: {sorted(done) if done else 'none'}\n")

    t_start = time.time()
    for rank, (idx, sc, h) in enumerate(scenarios, start=1):
        if rank in done:
            print(f"  [skip] scen #{rank}")
            continue
        ts = time.time()
        cells = run_2x2(spec, obs, sc, axis=axis, seed_offset=rank * 100,
                        base_seed=base_seed)
        dt = time.time() - ts
        ma = cells["matched-aware"]["final_mean"]
        mn = cells["matched-naive"]["final_mean"]
        mia = cells["mismatched-aware"]["final_mean"]
        min_ = cells["mismatched-naive"]["final_mean"]
        ma_sd = cells["matched-aware"]["final_sd"]
        mia_sd = cells["mismatched-aware"]["final_sd"]
        min_sd = cells["mismatched-naive"]["final_sd"]
        sub_gap = ma - mia
        aware_gap = mia - min_
        trap_ok = ma > mia > min_
        mp = sc.truth.get("MainPower", "?")
        record = {
            "rank": rank, "pool_idx": idx, "hardness": h,
            "mainpower": mp,
            "matched_aware": ma, "matched_aware_sd": ma_sd,
            "matched_naive": mn,
            "mismatched_aware": mia, "mismatched_aware_sd": mia_sd,
            "mismatched_naive": min_, "mismatched_naive_sd": min_sd,
            "sub_gap": sub_gap, "aware_gap": aware_gap, "trap_ok": trap_ok,
            "actions_ma": cells["matched-aware"]["actions_mean"],
            "actions_mia": cells["mismatched-aware"]["actions_mean"],
            "actions_min": cells["mismatched-naive"]["actions_mean"],
            "seconds": dt,
        }
        data["results"].append(record)
        save_partial(out_path, data)
        print(f"  scen #{rank} (h={h}, mp={mp:5s}): "
              f"m-a={ma:.3f}  mm-a={mia:.3f}±{mia_sd:.3f}  mm-n={min_:.3f}±{min_sd:.3f}  "
              f"| sub={sub_gap:+.3f} aware={aware_gap:+.3f}  "
              f"trap={'YES' if trap_ok else 'no '}  [{dt:.1f}s, total {time.time()-t_start:.1f}s]")

    R = data["results"]
    if not R:
        return
    print("\n" + "=" * 100)
    print(f"domain17 {axis.upper()}-AXIS AGGREGATE across {len(R)} hard scenarios (THD={THD})")
    print("=" * 100)
    ma = np.array([r["matched_aware"] for r in R])
    mia = np.array([r["mismatched_aware"] for r in R])
    min_ = np.array([r["mismatched_naive"] for r in R])
    print(f"  matched-aware    : {ma.mean():.3f}  SD={ma.std():.3f}")
    print(f"  mismatched-aware : {mia.mean():.3f}  SD={mia.std():.3f}")
    print(f"  mismatched-naive : {min_.mean():.3f}  SD={min_.std():.3f}")
    print(f"\n  substrate gap : {ma.mean()-mia.mean():+.3f}")
    print(f"  aware gap     : {mia.mean()-min_.mean():+.3f}")
    sg = np.array([r["sub_gap"] for r in R])
    ag = np.array([r["aware_gap"] for r in R])
    print(f"\n  Substrate gap: mean {sg.mean():+.3f}, "
          f"pos {sum(g>0 for g in sg)}/{len(R)}, range [{sg.min():+.3f}, {sg.max():+.3f}]")
    print(f"  Aware gap    : mean {ag.mean():+.3f}, "
          f"pos {sum(g>0 for g in ag)}/{len(R)}, range [{ag.min():+.3f}, {ag.max():+.3f}]")
    if len(R) >= 2 and sg.std(ddof=1) > 0:
        print(f"  t(sub_gap != 0)   = {sg.mean()/(sg.std(ddof=1)/np.sqrt(len(sg))):.2f} df={len(R)-1}")
    if len(R) >= 2 and ag.std(ddof=1) > 0:
        print(f"  t(aware_gap != 0) = {ag.mean()/(ag.std(ddof=1)/np.sqrt(len(ag))):.2f} df={len(R)-1}")


if __name__ == "__main__":
    main()

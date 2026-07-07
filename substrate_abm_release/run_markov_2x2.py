"""
run_markov_2x2.py
=================
Markov-axis 2x2 on the same 8 hard scenarios as run_multiscen_2x2.py at THD=0.55.
Composition: matched (all BBN) vs mismatched (1 Markov + 2 BBN).

This is the second substrate axis: Markov = INFERENTIAL loss (drop non-dominant
parents), distinct from tree = REPRESENTATIONAL loss (crisp argmax, no joint).
Together with the tree results, this completes the substrate manipulation.

Refined-mechanism prediction (see MULTISCEN_REAL_STATUS.md):
    aware_gap on Markov axis << aware_gap on tree axis, because Markov's output
    is probabilistic like BBN, so substrate-driven L1/L2 divergence is smaller,
    so aware vs naive doesn't differ much in conflict-detection.
    Substrate cost may still be positive if Markov's approximation is lossy on
    multi-parent nodes.
"""
from __future__ import annotations
import json
import os
import pickle
import time
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from experiment import TeamConfig, run_many


IN_PATH = "./results/hard_scenarios.pkl"
OUT_PATH = "./results/markov_2x2_results.json"
THD = 0.55
N_RUNS = 40
BASE_SEED = 800000


def run_cell(spec, obs, scenario, engines, projection, n_runs, seed):
    tc = TeamConfig(engines=engines, projection_mode=projection,
                    charitable=False, thd=THD, gamma=0.95,
                    max_actions=100, idle_limit=5)
    return run_many(spec, obs, scenario, tc, n_runs=n_runs, seed=seed)


def run_2x2(spec, obs, scenario, seed_offset):
    cells = {
        "matched-aware":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},      projection="aware"),
        "matched-naive":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},      projection="naive"),
        "mismatched-aware": dict(engines={"A":"markov","B":"bbn","C":"bbn"},   projection="aware"),
        "mismatched-naive": dict(engines={"A":"markov","B":"bbn","C":"bbn"},   projection="naive"),
    }
    out = {}
    for i, (name, cfg) in enumerate(cells.items()):
        r = run_cell(spec, obs, scenario, cfg["engines"], cfg["projection"],
                     n_runs=N_RUNS, seed=BASE_SEED + seed_offset + i)
        out[name] = {"final_mean": r["final_mean"], "final_sd": r["final_sd"],
                     "actions_mean": r["actions_mean"]}
    return out


def load_partial():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {"thd": THD, "n_runs": N_RUNS, "results": [], "axis": "markov"}


def save_partial(data):
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    with open(IN_PATH, "rb") as f:
        payload = pickle.load(f)
    scenarios = payload["scenarios"]
    print(f"Markov-axis 2x2 on {len(scenarios)} hard scenarios, THD={THD}, N={N_RUNS}\n")

    data = load_partial()
    done = {r["rank"] for r in data["results"]}
    print(f"Already done: {sorted(done) if done else 'none'}\n")

    t_start = time.time()
    for rank, (idx, sc, h) in enumerate(scenarios, start=1):
        if rank in done:
            print(f"  [skip] scen #{rank}")
            continue
        ts = time.time()
        cells = run_2x2(spec, obs, sc, seed_offset=rank * 100)
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
        bv = sc.truth.get("BatVolt", "?")
        record = {
            "rank": rank, "pool_idx": idx, "hardness": h, "batvolt": bv,
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
        save_partial(data)
        print(f"  scen #{rank} (h={h}, bv={bv:5s}): "
              f"m-a={ma:.3f}  mm-a={mia:.3f}±{mia_sd:.3f}  mm-n={min_:.3f}±{min_sd:.3f}  "
              f"| sub={sub_gap:+.3f} aware={aware_gap:+.3f}  "
              f"trap={'YES' if trap_ok else 'no '}  [{dt:.1f}s, total {time.time()-t_start:.1f}s]")

    R = data["results"]
    if not R:
        return
    print("\n" + "=" * 100)
    print(f"MARKOV-AXIS AGGREGATE across {len(R)} hard scenarios (THD={THD})")
    print("=" * 100)
    ma = np.array([r["matched_aware"] for r in R])
    mia = np.array([r["mismatched_aware"] for r in R])
    min_ = np.array([r["mismatched_naive"] for r in R])
    print(f"  matched-aware    : {ma.mean():.3f}  SD_across_scen = {ma.std():.3f}")
    print(f"  mismatched-aware : {mia.mean():.3f}  SD = {mia.std():.3f}")
    print(f"  mismatched-naive : {min_.mean():.3f}  SD = {min_.std():.3f}")
    print(f"\n  substrate gap : {ma.mean()-mia.mean():+.3f}")
    print(f"  aware gap     : {mia.mean()-min_.mean():+.3f}")
    sg = np.array([r["sub_gap"] for r in R])
    ag = np.array([r["aware_gap"] for r in R])
    print(f"\n  Substrate gap : mean {sg.mean():+.3f}, positive {sum(g>0 for g in sg)}/{len(R)}, "
          f"range [{sg.min():+.3f}, {sg.max():+.3f}]")
    print(f"  Aware gap     : mean {ag.mean():+.3f}, positive {sum(g>0 for g in ag)}/{len(R)}, "
          f"range [{ag.min():+.3f}, {ag.max():+.3f}]")
    if len(R) >= 2:
        ag_t = ag.mean() / (ag.std(ddof=1)/np.sqrt(len(ag))) if ag.std(ddof=1) > 0 else float("nan")
        sg_t = sg.mean() / (sg.std(ddof=1)/np.sqrt(len(sg))) if sg.std(ddof=1) > 0 else float("nan")
        print(f"  t(aware_gap != 0) = {ag_t:.2f} df={len(R)-1}")
        print(f"  t(sub_gap != 0)   = {sg_t:.2f} df={len(R)-1}")

    # by BatVolt
    print(f"\n  Aware gap by BatVolt (Markov axis):")
    for bvs in ["dead", "weak"]:
        subset = [r["aware_gap"] for r in R if r["batvolt"] == bvs]
        if subset:
            print(f"    {bvs} (n={len(subset)}): mean {np.mean(subset):+.3f}, "
                  f"range [{min(subset):+.3f}, {max(subset):+.3f}]")


if __name__ == "__main__":
    main()

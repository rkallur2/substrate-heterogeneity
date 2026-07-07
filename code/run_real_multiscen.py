"""
run_real_multiscen.py
=====================
Multi-scenario replication of the THD=0.55 real-network 2x2. Samples a large pool
of Starts=False scenarios on the real Kanno network, takes the top K_HARD by
hardness (# non-nominal-state nodes), and runs the full 2x2 on each with N runs
per cell. Reports per-scenario results, aggregate stats, and Trap-ordering
hold/break tallies.

Purpose: is the THD=0.55 result from run_real_2x2_hard.py scenario-specific?
"""
from __future__ import annotations
import random
import time
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from simulation import sample_scenario
from experiment import TeamConfig, run_many


POOL_SIZE = 300         # sample this many Starts=False scenarios
K_HARD = 10             # keep top-K by hardness
POOL_SEED = 20260706    # deterministic scenario pool
BASE_RUN_SEED = 500000
N_RUNS = 40             # per cell
THD = 0.55


def hardness(spec, s):
    return sum(1 for n, v in s.truth.items() if v != spec.nodes[n][0])


def sample_pool(spec):
    """Sample POOL_SIZE Starts=False scenarios, deterministic given POOL_SEED."""
    rng = random.Random(POOL_SEED)
    got = []
    tries = 0
    while len(got) < POOL_SIZE and tries < POOL_SIZE * 20:
        tries += 1
        sc = sample_scenario(spec, rng, name=f"pool{len(got)}")
        if sc.truth["Starts"] == "False":
            got.append(sc)
    return got, tries


def run_cell(spec, obs, scenario, engines, projection, thd, n_runs, seed):
    tc = TeamConfig(engines=engines, projection_mode=projection,
                    charitable=False, thd=thd, gamma=0.95,
                    max_actions=100, idle_limit=5)
    return run_many(spec, obs, scenario, tc, n_runs=n_runs, seed=seed)


def run_2x2_one_scenario(spec, obs, scenario, thd, n_runs, seed_offset):
    cells = {
        "matched-aware":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="aware"),
        "matched-naive":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="naive"),
        "mismatched-aware": dict(engines={"A":"tree","B":"bbn","C":"bbn"},  projection="aware"),
        "mismatched-naive": dict(engines={"A":"tree","B":"bbn","C":"bbn"},  projection="naive"),
    }
    out = {}
    for i, (name, cfg) in enumerate(cells.items()):
        r = run_cell(spec, obs, scenario, cfg["engines"], cfg["projection"],
                     thd=thd, n_runs=n_runs, seed=BASE_RUN_SEED + seed_offset + i)
        out[name] = r
    return out


def main():
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    print(f"Real Kanno network: {len(spec.nodes)} nodes, 3 agents, THD={THD}\n")

    t0 = time.time()
    pool, tries = sample_pool(spec)
    print(f"Sampled {len(pool)} Starts=False scenarios in {tries} tries ({time.time()-t0:.1f}s)")

    # Hardness distribution
    hardnesses = [hardness(spec, s) for s in pool]
    from collections import Counter
    ct = Counter(hardnesses)
    print(f"Hardness distribution across pool: "
          f"mean={np.mean(hardnesses):.2f} SD={np.std(hardnesses):.2f} "
          f"min={min(hardnesses)} max={max(hardnesses)}")
    for h in sorted(ct.keys()):
        print(f"  h={h:2d}: {ct[h]:3d}")

    # Top-K by hardness (deterministic ordering: hardness DESC, then pool index ASC)
    ranked = sorted(enumerate(pool), key=lambda kv: (-hardness(spec, kv[1]), kv[0]))
    hard = [pair for pair in ranked[:K_HARD]]
    print(f"\nTop {K_HARD} scenarios by hardness:")
    for rank, (idx, sc) in enumerate(hard):
        h = hardness(spec, sc)
        faults = [f"{n}={sc.truth[n]}" for n in spec.nodes
                  if sc.truth[n] != spec.nodes[n][0]]
        print(f"  #{rank+1}: pool-idx={idx}, h={h}  faults: {', '.join(faults)}")

    # Run 2x2 on each
    print(f"\nRunning 2x2 at THD={THD}, N={N_RUNS} runs/cell, on {K_HARD} scenarios...")
    print("=" * 100)
    all_results = []
    t_start = time.time()
    for rank, (idx, sc) in enumerate(hard):
        h = hardness(spec, sc)
        ts = time.time()
        r = run_2x2_one_scenario(spec, obs, sc, thd=THD, n_runs=N_RUNS,
                                 seed_offset=rank * 100)
        dt = time.time() - ts
        ma = r["matched-aware"]["final_mean"]
        mn = r["matched-naive"]["final_mean"]
        mia = r["mismatched-aware"]["final_mean"]
        min_ = r["mismatched-naive"]["final_mean"]
        ma_sd = r["matched-aware"]["final_sd"]
        mia_sd = r["mismatched-aware"]["final_sd"]
        min_sd = r["mismatched-naive"]["final_sd"]
        sub_gap = ma - mia
        aware_gap = mia - min_
        trap_ok = ma > mia > min_
        print(f"  scen #{rank+1} (h={h:2d}, idx={idx:3d}): "
              f"m-a={ma:.3f}±{ma_sd:.3f}  m-n={mn:.3f}  mm-a={mia:.3f}±{mia_sd:.3f}  "
              f"mm-n={min_:.3f}±{min_sd:.3f}  | sub={sub_gap:+.3f} aware={aware_gap:+.3f}  "
              f"trap={'YES' if trap_ok else 'no '}  [{dt:.1f}s]")
        all_results.append({
            "rank": rank + 1, "idx": idx, "hardness": h,
            "ma": ma, "ma_sd": ma_sd, "mn": mn,
            "mia": mia, "mia_sd": mia_sd,
            "min": min_, "min_sd": min_sd,
            "sub_gap": sub_gap, "aware_gap": aware_gap, "trap_ok": trap_ok,
            "actions_ma": r["matched-aware"]["actions_mean"],
            "actions_mia": r["mismatched-aware"]["actions_mean"],
            "actions_min": r["mismatched-naive"]["actions_mean"],
        })
    print(f"\n  Total: {time.time()-t_start:.1f}s")

    # Aggregate
    print("\n" + "=" * 100)
    print(f"AGGREGATE across {K_HARD} hard scenarios (THD={THD}, N={N_RUNS} runs/cell/scenario)")
    print("=" * 100)
    mean_ma = np.mean([r["ma"] for r in all_results])
    mean_mn = np.mean([r["mn"] for r in all_results])
    mean_mia = np.mean([r["mia"] for r in all_results])
    mean_min = np.mean([r["min"] for r in all_results])
    sd_ma_across_scen = np.std([r["ma"] for r in all_results])
    sd_mia_across_scen = np.std([r["mia"] for r in all_results])
    sd_min_across_scen = np.std([r["min"] for r in all_results])
    print(f"  matched-aware    : mean = {mean_ma:.3f}  (SD across scenarios = {sd_ma_across_scen:.3f})")
    print(f"  matched-naive    : mean = {mean_mn:.3f}")
    print(f"  mismatched-aware : mean = {mean_mia:.3f}  (SD across scenarios = {sd_mia_across_scen:.3f})")
    print(f"  mismatched-naive : mean = {mean_min:.3f}  (SD across scenarios = {sd_min_across_scen:.3f})")
    print(f"\n  Aggregate substrate gap  = {mean_ma - mean_mia:+.3f}")
    print(f"  Aggregate aware-naive gap = {mean_mia - mean_min:+.3f}")
    n_trap_ok = sum(1 for r in all_results if r["trap_ok"])
    print(f"\n  Trap ordering held in {n_trap_ok}/{K_HARD} scenarios")
    # per-scenario gap distribution
    sub_gaps = [r["sub_gap"] for r in all_results]
    aware_gaps = [r["aware_gap"] for r in all_results]
    print(f"  Substrate gap : mean = {np.mean(sub_gaps):+.3f}  SD = {np.std(sub_gaps):.3f}  "
          f"range = [{min(sub_gaps):+.3f}, {max(sub_gaps):+.3f}]")
    print(f"  Aware gap     : mean = {np.mean(aware_gaps):+.3f}  SD = {np.std(aware_gaps):.3f}  "
          f"range = [{min(aware_gaps):+.3f}, {max(aware_gaps):+.3f}]")

    # sign counts
    n_sub_positive = sum(1 for g in sub_gaps if g > 0)
    n_aware_positive = sum(1 for g in aware_gaps if g > 0)
    print(f"  Substrate gap > 0 in {n_sub_positive}/{K_HARD} scenarios")
    print(f"  Aware gap > 0 in {n_aware_positive}/{K_HARD} scenarios")

    # simple statistical test on aware_gap != 0 (one-sample t-test)
    ag_arr = np.array(aware_gaps)
    if ag_arr.std() > 0:
        t_stat = ag_arr.mean() / (ag_arr.std(ddof=1) / np.sqrt(len(ag_arr)))
        print(f"  Aware gap: t-stat vs 0 (df={len(ag_arr)-1}) = {t_stat:.3f}")
    sg_arr = np.array(sub_gaps)
    if sg_arr.std() > 0:
        t_stat = sg_arr.mean() / (sg_arr.std(ddof=1) / np.sqrt(len(sg_arr)))
        print(f"  Substrate gap: t-stat vs 0 (df={len(sg_arr)-1}) = {t_stat:.3f}")


if __name__ == "__main__":
    main()

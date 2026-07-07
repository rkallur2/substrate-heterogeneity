"""
run_real_2x2_hard.py
====================
2x2 on the real network, SHARED baseline (all agent1.dne), on the HARDEST of 30
Starts=False scenarios (10 faulty nodes -- a real cooperative diagnosis problem),
across THD sweep. Same seed protocol as run_real_2x2.py.
"""
from __future__ import annotations
import random
import time
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from simulation import sample_scenario
from experiment import TeamConfig, run_many
from bbn_engine import UNCERTAIN


def find_hard_scenario(spec, target=("Starts", "False"), K=30, seed=42):
    """Sample K scenarios; return the one with the most non-nominal-state nodes."""
    rng = random.Random(seed)
    node, want = target
    scenarios = []
    while len(scenarios) < K:
        sc = sample_scenario(spec, rng, name=f"s{len(scenarios)}")
        if sc.truth[node] == want:
            scenarios.append(sc)

    def hardness(s):
        return sum(1 for n, v in s.truth.items() if v != spec.nodes[n][0])

    idx = int(np.argmax([hardness(s) for s in scenarios]))
    hard = scenarios[idx]
    print(f"Hardest of {K} scenarios: idx={idx}, {hardness(hard)} faulty nodes")
    return hard


def run_2x2(spec, obs, scenario, thd, n_runs=40, base_seed=1000):
    cells = {
        "matched-aware":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="aware"),
        "matched-naive":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="naive"),
        "mismatched-aware": dict(engines={"A":"tree","B":"bbn","C":"bbn"},  projection="aware"),
        "mismatched-naive": dict(engines={"A":"tree","B":"bbn","C":"bbn"},  projection="naive"),
    }
    results = {}
    for i, (name, cfg) in enumerate(cells.items()):
        t0 = time.time()
        tc = TeamConfig(engines=cfg["engines"], projection_mode=cfg["projection"],
                        charitable=False, thd=thd, gamma=0.95,
                        max_actions=100, idle_limit=5)
        r = run_many(spec, obs, scenario, tc, n_runs=n_runs, seed=base_seed + i)
        dt = time.time() - t0
        results[name] = r
        print(f"  THD={thd:.2f} {name:20s} final={r['final_mean']:.3f} ±{r['final_sd']:.3f}  "
              f"actions={r['actions_mean']:5.1f}  [{dt:.1f}s]")
    return results


def main():
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    print(f"Real network: {len(spec.nodes)} nodes, 3 agents\n")

    hard = find_hard_scenario(spec, K=30, seed=42)
    print("Faulty nodes in this scenario:")
    for n, v in hard.truth.items():
        if v != spec.nodes[n][0]:
            print(f"  {n:12s} = {v}")
    print()

    all_results = {}
    for thd in [0.55, 0.65, 0.75, 0.85]:
        print(f"\n---- 2x2 at THD={thd} ----")
        r = run_2x2(spec, obs, hard, thd=thd, n_runs=40)
        all_results[thd] = r

    # Table
    print("\n" + "=" * 100)
    print("HARDEST-SCENARIO 2x2 across THDs (final accuracy ± SD)")
    print("=" * 100)
    hdr = f"{'THD':>6} | " + " | ".join(f"{k:<20}" for k in all_results[0.55].keys())
    print(hdr)
    print("-" * len(hdr))
    for thd, results in all_results.items():
        row = f"{thd:>6.2f} | " + " | ".join(
            f"{r['final_mean']:.3f} ±{r['final_sd']:.3f}      " for r in results.values()
        )
        print(row)

    # Trap ordering per THD
    print("\nTrap ordering check per THD (want: matched-aware > mismatched-aware > mismatched-naive):")
    for thd, results in all_results.items():
        ma = results["matched-aware"]["final_mean"]
        mn = results["matched-naive"]["final_mean"]
        mia = results["mismatched-aware"]["final_mean"]
        min_ = results["mismatched-naive"]["final_mean"]
        ord_ok = ma > mia > min_
        awn_gap = mia - min_
        sub_gap = ma - mia
        print(f"  THD={thd:.2f}: matched={ma:.3f} (m-naive={mn:.3f})  mismatched-aware={mia:.3f}  "
              f"mismatched-naive={min_:.3f}  | sub_gap={sub_gap:+.3f}  aware_gap={awn_gap:+.3f}  "
              f"trap_ok={ord_ok}")


if __name__ == "__main__":
    main()

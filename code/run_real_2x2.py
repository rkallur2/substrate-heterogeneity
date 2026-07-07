"""
run_real_2x2.py
===============
The headline result: substrate 2x2 (matched vs 1-tree-mismatched, aware vs naive)
on Kanno's REAL car-diagnosis network under the SHARED-FILE baseline (all agents
use agent1.dne, per the faithful "identical agents" replication).

Task #1 from REAL_NETWORK_WIRING_STATUS.md next-session list.
Needs no Kanno reply -- uses the anchor configuration.

Design:
- One shared Starts=False scenario across all cells (pinned by seed) so all cells
  solve the same diagnostic problem; cross-condition scenario noise is eliminated.
- Composition:
    matched      = {A: bbn, B: bbn, C: bbn}
    mismatched   = {A: tree, B: bbn, C: bbn}          # 1 tree + 2 BBN
- 2x2:  {matched, mismatched} x {aware, naive}
- 40 runs per cell (matches domain17 table).
- Ran first at THD=0.55; if the baseline is ceiling'd (real-network parametric
  validation hit 0.979 at THD=0.55), sweep upward to find the mid-range window.

The prediction (Anthropomorphization Trap):
    matched-aware ~= matched-naive  > mismatched-aware  > mismatched-naive
"""
from __future__ import annotations
import random
import time
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from simulation import sample_scenario
from experiment import TeamConfig, run_many


def pinned_scenario(spec, target_state=("Starts", "False"), max_tries=200, seed=0):
    """Sample scenarios until the target node hits the target state; deterministic
    given seed. Returns the first matching Scenario."""
    rng = random.Random(seed)
    node, want = target_state
    for i in range(max_tries):
        sc = sample_scenario(spec, rng, name=f"real_starts_{want}_{i}")
        if sc.truth[node] == want:
            return sc, i + 1
    raise RuntimeError(f"No {node}={want} scenario in {max_tries} tries")


def run_cell(spec, obs, scenario, engines, projection, thd, n_runs=40, seed=0):
    tc = TeamConfig(engines=engines, projection_mode=projection,
                    charitable=False, thd=thd, gamma=0.95,
                    max_actions=100, idle_limit=5)
    return run_many(spec, obs, scenario, tc, n_runs=n_runs, seed=seed)


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
        r = run_cell(spec, obs, scenario, cfg["engines"], cfg["projection"],
                     thd=thd, n_runs=n_runs, seed=base_seed + i)
        dt = time.time() - t0
        results[name] = r
        print(f"  {name:20s} final={r['final_mean']:.3f} ±{r['final_sd']:.3f}  "
              f"actions={r['actions_mean']:5.1f}  [{dt:.1f}s]")
    return results


def print_table(thd, results):
    print(f"\n=== THD = {thd} ===")
    print(f"{'condition':<22} {'final acc':>12} {'actions':>10}")
    print("-" * 46)
    for name, r in results.items():
        print(f"{name:<22} {r['final_mean']:>6.3f} ±{r['final_sd']:.3f}  {r['actions_mean']:>10.1f}")
    # Trap check
    ma = results["matched-aware"]["final_mean"]
    mia = results["mismatched-aware"]["final_mean"]
    min_ = results["mismatched-naive"]["final_mean"]
    print(f"\nTrap ordering check (want matched > mismatched-aware > mismatched-naive):")
    print(f"  matched-aware       = {ma:.3f}")
    print(f"  mismatched-aware    = {mia:.3f}   (gap from matched: {ma-mia:+.3f})")
    print(f"  mismatched-naive    = {min_:.3f}   (gap from aware:   {mia-min_:+.3f})")
    ord_ok = ma > mia > min_
    print(f"  ordering satisfied: {ord_ok}")


def main():
    print("Loading real Kanno network (agent1.dne, shared-file baseline)...")
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    print(f"  {len(spec.nodes)} nodes, {len(obs)} agents\n")

    print("Sampling Starts=False scenario (seed=0)...")
    scenario, tries = pinned_scenario(spec, ("Starts", "False"), seed=0)
    print(f"  Found on try {tries}. Truth (unobservable targets):")
    for n in ["Starts", "PlugVolt", "Starter", "Dist", "SpkQual", "Air", "Timing"]:
        if n in scenario.truth:
            print(f"    {n} = {scenario.truth[n]}")
    print()

    # Sweep THDs. Start at 0.55 (matches domain17); go up to find mid-range if ceilinged.
    all_results = {}
    for thd in [0.55, 0.65, 0.75, 0.85]:
        print(f"\n---- Running 2x2 at THD={thd} ----")
        r = run_2x2(spec, obs, scenario, thd=thd, n_runs=40)
        all_results[thd] = r
        print_table(thd, r)

    # Summary across THDs
    print("\n" + "=" * 72)
    print("SUMMARY: final accuracy across THDs")
    print("=" * 72)
    print(f"{'THD':>6} | " + " | ".join(f"{k:<20}" for k in all_results[0.55].keys()))
    print("-" * 100)
    for thd, results in all_results.items():
        row = f"{thd:>6.2f} | " + " | ".join(
            f"{r['final_mean']:.3f} ±{r['final_sd']:.3f}      " for r in results.values()
        )
        print(row)


if __name__ == "__main__":
    main()

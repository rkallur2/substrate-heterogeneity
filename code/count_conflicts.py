"""
count_conflicts.py
==================
Direct test of the conflict-detection mechanism proposed in MULTISCEN_REAL_STATUS.md.

Claim: in AWARE mismatched teams, BBN agents' L2 models of the tree teammate are
TREE engines, which produce diagnoses that differ from the BBN L1 (same evidence,
different substrate). Conflicts fire, cooperation triggers.
In NAIVE teams, BBN agents' L2 models of the tree teammate are BBN engines, so
L1 ~= L2(tree, BBN). Fewer/no substrate-driven conflicts perceived.

Test: for each Kanno hard scenario, build mismatched-aware and mismatched-naive
teams, run observation phase only, count conflicts perceived by each agent
BEFORE cooperation. Break down:
  - total conflicts across the team
  - conflicts involving the tree teammate (agent A) vs conflicts among BBN agents
  - by scenario, correlate with the naive-stall pattern

Prediction:
  - aware total > naive total (mechanism)
  - aware A-involved > naive A-involved (specifically the substrate-driven kind)
  - BBN-among-BBN conflict counts similar in aware and naive (baseline)
  - In dead-battery scenarios where naive-stall was observed (S3, S6, S8),
    naive should show near-zero total conflicts.
"""
from __future__ import annotations
import json
import pickle
import random
from collections import Counter
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from experiment import TeamConfig, build_team


IN_PATH = "/home/claude/substrate_abm/hard_scenarios.pkl"
OUT_PATH = "/home/claude/substrate_abm/conflict_counts.json"
THD = 0.55
N_RUNS = 20   # different rng shuffle each time; conflicts depend on observation order (all get observed, but nothing else varies here so this is not strictly needed — but keep for robustness)


def instrument_scenario(spec, obs, scenario, projection, seed):
    tc = TeamConfig(engines={"A":"tree","B":"bbn","C":"bbn"},
                    projection_mode=projection, charitable=False,
                    thd=THD, gamma=0.95, max_actions=100, idle_limit=5)
    rng = random.Random(seed)
    agents = build_team(spec, obs, tc)

    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])

    # snapshot conflicts perceived by each agent
    per_agent = {}
    for a in agents:
        cfs = a.find_conflicts()  # list of (node, type, teammate)
        # separate: conflicts A perceives with the tree agent vs among BBN
        conflicts_with_tree = [c for c in cfs if c[2] == "A"] if a.name != "A" else []
        conflicts_among_bbn = [c for c in cfs if c[2] != "A"] if a.name != "A" else cfs
        per_agent[a.name] = {
            "total": len(cfs),
            "with_A": len(conflicts_with_tree),
            "not_with_A": len(conflicts_among_bbn),
        }
    total = sum(pa["total"] for pa in per_agent.values())
    total_with_A = sum(pa["with_A"] for pa in per_agent.values())
    total_not_with_A = sum(pa["not_with_A"] for pa in per_agent.values())
    return {
        "per_agent": per_agent,
        "total": total,
        "total_with_A": total_with_A,
        "total_not_with_A": total_not_with_A,
    }


def main():
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    with open(IN_PATH, "rb") as f:
        pool = pickle.load(f)["scenarios"]

    print(f"Direct conflict-count instrumentation on {len(pool)} Kanno hard scenarios, THD={THD}\n")

    all_results = []
    for rank, (idx, sc, h) in enumerate(pool, start=1):
        bv = sc.truth.get("BatVolt", "?")

        aware_runs = [instrument_scenario(spec, obs, sc, "aware", seed=800000 + rank*100 + i)
                      for i in range(N_RUNS)]
        naive_runs = [instrument_scenario(spec, obs, sc, "naive", seed=800000 + rank*100 + 50 + i)
                      for i in range(N_RUNS)]

        def agg(runs, key):
            return float(np.mean([r[key] for r in runs]))

        aware_tot = agg(aware_runs, "total")
        aware_wA  = agg(aware_runs, "total_with_A")
        aware_nA  = agg(aware_runs, "total_not_with_A")
        naive_tot = agg(naive_runs, "total")
        naive_wA  = agg(naive_runs, "total_with_A")
        naive_nA  = agg(naive_runs, "total_not_with_A")

        rec = {
            "rank": rank, "pool_idx": idx, "hardness": h, "batvolt": bv,
            "aware": {"total": aware_tot, "with_A": aware_wA, "not_with_A": aware_nA},
            "naive": {"total": naive_tot, "with_A": naive_wA, "not_with_A": naive_nA},
        }
        all_results.append(rec)
        print(f"  #{rank} (h={h}, bv={bv:5s}):  "
              f"AWARE total={aware_tot:5.1f}  with-A={aware_wA:5.1f}  not-with-A={aware_nA:5.1f}  |  "
              f"NAIVE total={naive_tot:5.1f}  with-A={naive_wA:5.1f}  not-with-A={naive_nA:5.1f}")

    with open(OUT_PATH, "w") as f:
        json.dump({"thd": THD, "n_runs": N_RUNS, "results": all_results}, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    # aggregates by BatVolt
    print("\n" + "=" * 100)
    print("AGGREGATES BY BATVOLT STATE (mean conflicts across scenarios in that group)")
    print("=" * 100)
    for bvs in ["dead", "weak"]:
        subset = [r for r in all_results if r["batvolt"] == bvs]
        if not subset:
            continue
        aw_tot = np.mean([r["aware"]["total"] for r in subset])
        aw_wA  = np.mean([r["aware"]["with_A"] for r in subset])
        aw_nA  = np.mean([r["aware"]["not_with_A"] for r in subset])
        na_tot = np.mean([r["naive"]["total"] for r in subset])
        na_wA  = np.mean([r["naive"]["with_A"] for r in subset])
        na_nA  = np.mean([r["naive"]["not_with_A"] for r in subset])
        print(f"\n  {bvs} scenarios (n={len(subset)}):")
        print(f"    AWARE: total={aw_tot:.1f}  with-A={aw_wA:.1f}  not-with-A={aw_nA:.1f}")
        print(f"    NAIVE: total={na_tot:.1f}  with-A={na_wA:.1f}  not-with-A={na_nA:.1f}")
        print(f"    Difference (aware - naive):  total={aw_tot-na_tot:+.1f}  "
              f"with-A={aw_wA-na_wA:+.1f}  not-with-A={aw_nA-na_nA:+.1f}")

    # overall
    print("\n" + "-" * 100)
    print("OVERALL (all 8 scenarios):")
    aw_tot = np.mean([r["aware"]["total"] for r in all_results])
    aw_wA  = np.mean([r["aware"]["with_A"] for r in all_results])
    aw_nA  = np.mean([r["aware"]["not_with_A"] for r in all_results])
    na_tot = np.mean([r["naive"]["total"] for r in all_results])
    na_wA  = np.mean([r["naive"]["with_A"] for r in all_results])
    na_nA  = np.mean([r["naive"]["not_with_A"] for r in all_results])
    print(f"  AWARE: total={aw_tot:.1f}  with-A={aw_wA:.1f}  not-with-A={aw_nA:.1f}")
    print(f"  NAIVE: total={na_tot:.1f}  with-A={na_wA:.1f}  not-with-A={na_nA:.1f}")
    print(f"  Difference: total={aw_tot-na_tot:+.1f}  with-A={aw_wA-na_wA:+.1f}  not-with-A={aw_nA-na_nA:+.1f}")


if __name__ == "__main__":
    main()

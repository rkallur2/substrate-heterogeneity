"""
instrument_tree_correctness.py
==============================
Test the mechanism proposed in MULTISCEN_REAL_STATUS.md:
    aware > naive when the tree agent's crisp diagnoses tend to be RIGHT;
    naive > aware when the tree agent's crisp diagnoses tend to be WRONG.

Measurement: run a mismatched-aware team on each of the 8 hard scenarios at
THD=0.55, snapshot the tree agent's L1 diagnoses at two points -- after the
observation phase (its "communication-time state") and at end-of-run. Compute:
  - commit rate  = fraction of 17 nodes where tree's diagnosis != UNCERTAIN
  - correctness  = fraction of COMMITTED nodes where tree's diagnosis == truth

Cross-reference with the aware_gap already measured, look for the predicted
positive correlation.

Also runs a mismatched-NAIVE version as a control -- the tree agent's own L1
correctness shouldn't depend on other agents' projection modes, but this
verifies the assumption.
"""
from __future__ import annotations
import json
import pickle
import random
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from experiment import TeamConfig, build_team
from simulation import node_importance
from bbn_engine import UNCERTAIN


IN_HARD_PATH = "./results/hard_scenarios.pkl"
IN_MULTISCEN_PATH = "./results/multiscen_results.json"
OUT_PATH = "./results/tree_correctness.json"
THD = 0.55
N_RUNS = 20   # per scenario
BASE_SEED = 700000


def tree_snapshot(agents, thd, truth):
    """Return (commit_rate, correctness) for the tree agent (agent A)."""
    tree = agents[0]
    assert tree.M.__class__.__name__ == "DecisionTreeEngine", tree.M.__class__.__name__
    committed = 0
    correct = 0
    n = 0
    for node in tree.spec.nodes:
        n += 1
        d = tree.M.diagnosis(node, thd)
        if d == UNCERTAIN:
            continue
        committed += 1
        if d == truth[node]:
            correct += 1
    commit_rate = committed / n if n else 0.0
    correctness = (correct / committed) if committed else float("nan")
    return commit_rate, correctness, committed, correct


def instrumented_run(spec, obs, scenario, projection, thd, seed):
    """Run a mismatched (tree/BBN/BBN) team through a full simulation, snapshot
    the tree agent's L1 state post-observation and at end-of-run."""
    from experiment import _resolve, engine_kind

    tc = TeamConfig(engines={"A":"tree","B":"bbn","C":"bbn"},
                    projection_mode=projection, charitable=False,
                    thd=thd, gamma=0.95, max_actions=100, idle_limit=5)
    rng = random.Random(seed)
    importance = node_importance(spec)

    agents = build_team(spec, obs, tc)
    by_name = {a.name: a for a in agents}

    # observation phase
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])

    # snapshot post-observation
    post_obs = tree_snapshot(agents, thd, scenario.truth)

    # cooperation phase (replicate the key logic from experiment.run_once)
    def diag_state(ags):
        return tuple(tuple(a.M.diagnosis(n, tc.thd) for n in spec.nodes) for a in ags)

    idle = 0
    last = diag_state(agents)
    actions = 0
    max_actions = tc.max_actions
    while actions < max_actions:
        if all(not a.find_conflicts() for a in agents):
            break
        actor = rng.choice(agents)
        conflicts = actor.find_conflicts()
        if not conflicts:
            continue
        node, ctype, mate = max(conflicts, key=lambda c: importance[c[0]])
        _resolve(actor, by_name, importance, node, ctype, mate, scenario, tc)
        actions += 1
        now = diag_state(agents)
        if now == last:
            idle += 1
            if idle >= tc.idle_limit:
                break
        else:
            idle = 0
            last = now

    # snapshot final
    final = tree_snapshot(agents, thd, scenario.truth)

    # per-agent final accuracy
    per_agent = {}
    for a in agents:
        cor = sum(1 for n in spec.nodes if a.M.diagnosis(n, thd) == scenario.truth[n])
        per_agent[a.name] = cor / len(spec.nodes)

    return {
        "post_obs_commit_rate": post_obs[0],
        "post_obs_correctness": post_obs[1],
        "post_obs_committed": post_obs[2],
        "post_obs_correct":   post_obs[3],
        "final_commit_rate":  final[0],
        "final_correctness":  final[1],
        "final_committed":    final[2],
        "final_correct":      final[3],
        "actions": actions,
        "per_agent_final_acc": per_agent,
    }


def aggregate_across_runs(runs):
    """Average metrics across N runs of the same scenario."""
    keys_scalar = ["post_obs_commit_rate", "post_obs_correctness",
                   "post_obs_committed", "post_obs_correct",
                   "final_commit_rate", "final_correctness",
                   "final_committed", "final_correct", "actions"]
    out = {}
    for k in keys_scalar:
        vals = [r[k] for r in runs if not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[k + "_sd"] = float(np.std(vals)) if vals else float("nan")
    return out


def main():
    spec = make_spec_shared("agent1")
    obs = make_observability_real()
    with open(IN_HARD_PATH, "rb") as f:
        pool = pickle.load(f)["scenarios"]
    with open(IN_MULTISCEN_PATH) as f:
        multiscen = json.load(f)
    scen_meta = {r["rank"]: r for r in multiscen["results"]}

    print(f"Instrumenting tree correctness on {len(pool)} hard scenarios at THD={THD}, N={N_RUNS} runs\n")

    all_results = []
    for rank, (idx, sc, h) in enumerate(pool, start=1):
        bv = sc.truth.get("BatVolt", "?")
        # run N_RUNS mismatched-aware runs
        aware_runs = [instrumented_run(spec, obs, sc, projection="aware",
                                       thd=THD, seed=BASE_SEED + rank*1000 + i)
                      for i in range(N_RUNS)]
        aware_agg = aggregate_across_runs(aware_runs)
        # run N_RUNS mismatched-naive runs (control)
        naive_runs = [instrumented_run(spec, obs, sc, projection="naive",
                                       thd=THD, seed=BASE_SEED + rank*1000 + 500 + i)
                      for i in range(N_RUNS)]
        naive_agg = aggregate_across_runs(naive_runs)

        aware_gap = scen_meta[rank]["aware_gap"]

        rec = {
            "rank": rank, "pool_idx": idx, "hardness": h, "batvolt": bv,
            "aware_gap": aware_gap,
            "aware": aware_agg,
            "naive": naive_agg,
        }
        all_results.append(rec)
        print(f"  scen #{rank} (h={h}, bv={bv:5s}) aware_gap={aware_gap:+.3f}")
        print(f"    aware:  post-obs corr={aware_agg['post_obs_correctness']:.3f} "
              f"(commit={aware_agg['post_obs_commit_rate']:.3f})  |  "
              f"final corr={aware_agg['final_correctness']:.3f} "
              f"(commit={aware_agg['final_commit_rate']:.3f}) actions={aware_agg['actions']:.1f}")
        print(f"    naive:  post-obs corr={naive_agg['post_obs_correctness']:.3f} "
              f"(commit={naive_agg['post_obs_commit_rate']:.3f})  |  "
              f"final corr={naive_agg['final_correctness']:.3f} "
              f"(commit={naive_agg['final_commit_rate']:.3f}) actions={naive_agg['actions']:.1f}")

    # save
    with open(OUT_PATH, "w") as f:
        json.dump({"thd": THD, "n_runs": N_RUNS, "results": all_results}, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    # Correlations
    print("\n" + "=" * 100)
    print("MECHANISM TEST: does tree correctness predict aware_gap?")
    print("=" * 100)
    aware_gaps = [r["aware_gap"] for r in all_results]
    post_obs_corr = [r["aware"]["post_obs_correctness"] for r in all_results]
    final_corr = [r["aware"]["final_correctness"] for r in all_results]

    def pearson(x, y):
        x = np.array(x); y = np.array(y)
        if x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    r_post = pearson(post_obs_corr, aware_gaps)
    r_final = pearson(final_corr, aware_gaps)
    print(f"\n  Pearson r (post-obs tree correctness, aware_gap) = {r_post:+.3f}  (n={len(all_results)})")
    print(f"  Pearson r (final    tree correctness, aware_gap) = {r_final:+.3f}")

    print(f"\n  Post-observation tree correctness by BatVolt state (aware condition):")
    for bvs in ["dead", "weak"]:
        subset = [(r["aware"]["post_obs_correctness"], r["aware_gap"])
                  for r in all_results if r["batvolt"] == bvs]
        if subset:
            corrs = [s[0] for s in subset]
            gaps = [s[1] for s in subset]
            print(f"    {bvs} (n={len(subset)}): corr mean={np.mean(corrs):.3f}, "
                  f"range=[{min(corrs):.3f}, {max(corrs):.3f}]  |  "
                  f"gap mean={np.mean(gaps):+.3f}")

    print(f"\n  Final-state tree correctness by BatVolt state:")
    for bvs in ["dead", "weak"]:
        subset = [r["aware"]["final_correctness"]
                  for r in all_results if r["batvolt"] == bvs]
        if subset:
            print(f"    {bvs} (n={len(subset)}): mean={np.mean(subset):.3f}, "
                  f"range=[{min(subset):.3f}, {max(subset):.3f}]")


if __name__ == "__main__":
    main()

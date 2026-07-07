"""
diag_null_result.py
===================
Diagnose why the real-network 2x2 showed no Trap ordering.

Two candidate mechanisms:
1. The seed=0 scenario is trivial ("Starts=False but everything else Okay/good").
   Test: sample K=30 scenarios, measure hardness spread.
2. team_accuracy counts UNCERTAIN as wrong; at high THD, BBN abstains more than
   tree, giving the tree agent lucky-commitment credit.
   Test: run one cell and log per-agent UNCERTAIN counts + which engine commits
   correctly.
"""
from __future__ import annotations
import random
from collections import Counter
import numpy as np

from domain_real import make_spec_shared, make_observability_real
from simulation import sample_scenario
from bbn_engine import BBNEngine, UNCERTAIN
from experiment import TeamConfig, run_many, build_team


# ---------------------------------------------------------------- Hardness spread
def scenario_hardness(spec, scenario):
    """Number of nodes in a 'faulty' (non-first-state) truth value. Simple proxy."""
    # per-node state 0 = 'good/ok/nominal/strong/bright/True'-ish first entry
    # heuristic: count nodes whose truth != states[0]
    n_fault = 0
    for n, true in scenario.truth.items():
        states = spec.states[n] if hasattr(spec, "states") else list(spec.cpts[n].shape[-1:])
        # spec.nodes[n] is a list of state strings
        if isinstance(spec.nodes, dict):
            states = spec.nodes[n]
        if true != states[0]:
            n_fault += 1
    return n_fault


def hardness_spread(spec, target=("Starts", "False"), K=30):
    """Sample K scenarios with Starts=False, report hardness distribution."""
    rng = random.Random(42)
    node, want = target
    scenarios = []
    tries = 0
    while len(scenarios) < K and tries < 5000:
        tries += 1
        sc = sample_scenario(spec, rng, name=f"s{len(scenarios)}")
        if sc.truth[node] == want:
            scenarios.append(sc)
    hardness = [scenario_hardness(spec, s) for s in scenarios]
    print(f"Sampled {len(scenarios)} Starts=False scenarios in {tries} tries")
    print(f"Hardness (# non-nominal-state nodes out of {len(spec.nodes)}):")
    ct = Counter(hardness)
    for h in sorted(ct.keys()):
        print(f"  {h:2d} faulty nodes: {ct[h]:2d} scenarios  {'*' * ct[h]}")
    print(f"Mean hardness: {np.mean(hardness):.2f}, SD: {np.std(hardness):.2f}, "
          f"min: {min(hardness)}, max: {max(hardness)}")
    # what does the "hardest" scenario look like?
    idx_hard = int(np.argmax(hardness))
    print(f"\nHardest scenario (#{idx_hard}, {hardness[idx_hard]} faulty nodes):")
    for n, v in scenarios[idx_hard].truth.items():
        marker = "" if v == spec.nodes[n][0] else "  <-- fault"
        print(f"  {n:12s} = {v}{marker}")
    return scenarios, hardness


# ------------------------------------------------------ Commitment / abstention
def commitment_breakdown(spec, obs, scenario, engines, projection, thd, seed=100):
    """Run one team, one time, log per-agent commitment behavior."""
    from experiment import FACTORIES
    from embm_agent import EMBMAgent
    from simulation import team_accuracy, node_importance

    tc = TeamConfig(engines=engines, projection_mode=projection, charitable=False,
                    thd=thd, gamma=0.95, max_actions=100, idle_limit=5)
    rng = random.Random(seed)
    importance = node_importance(spec)

    # build team (same as build_team)
    agents = build_team(spec, obs, tc)
    # observation phase
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])

    # cooperation phase (mini version)
    from experiment import run_once
    r = run_once(spec, obs, scenario, tc, random.Random(seed + 1),
                 importance=importance)

    # inspect final agent states
    print(f"\n  Engines: {engines} | projection: {projection} | THD: {thd}")
    print(f"  Final acc: {r['final']:.3f}, actions: {r['actions']}")
    # rebuild for inspection at final state (or use previous agents if same)
    agents = build_team(spec, obs, tc)
    # re-run observation
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
    # simulate to end again (or just observe post-obs behavior)
    # for now, just show post-observation state
    print(f"  Post-observation commitments (before cooperation phase):")
    print(f"    {'node':12s} {'truth':10s} | " + " | ".join(f"{a.name:>10s}" for a in agents))
    n_uncertain = {a.name: 0 for a in agents}
    n_correct = {a.name: 0 for a in agents}
    n_wrong = {a.name: 0 for a in agents}
    for node in spec.nodes:
        row = f"    {node:12s} {scenario.truth[node]:10s} | "
        cells = []
        for a in agents:
            d = a.M.diagnosis(node, thd)
            if d == UNCERTAIN:
                cells.append(f"{'UNC':>10s}")
                n_uncertain[a.name] += 1
            elif d == scenario.truth[node]:
                cells.append(f"{d + '(+)':>10s}")
                n_correct[a.name] += 1
            else:
                cells.append(f"{d + '(-)':>10s}")
                n_wrong[a.name] += 1
        row += " | ".join(cells)
        print(row)
    print(f"  Per-agent tallies: ")
    for a in agents:
        eng_name = a.M.__class__.__name__
        print(f"    {a.name} ({eng_name}): correct={n_correct[a.name]}, "
              f"wrong={n_wrong[a.name]}, uncertain={n_uncertain[a.name]}")


def main():
    print("=" * 70)
    print("DIAGNOSTIC 1: scenario hardness spread")
    print("=" * 70)
    spec = make_spec_shared("agent1")
    scenarios, hardness = hardness_spread(spec, K=30)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: commitment/abstention behavior at THD=0.85")
    print("=" * 70)
    # use the hardest scenario for this
    obs = make_observability_real()
    idx_hard = int(np.argmax(hardness))
    hard_scenario = scenarios[idx_hard]
    print(f"Using scenario #{idx_hard} ({hardness[idx_hard]} faulty nodes)")

    commitment_breakdown(spec, obs, hard_scenario,
                         engines={"A":"bbn","B":"bbn","C":"bbn"},
                         projection="aware", thd=0.85)
    commitment_breakdown(spec, obs, hard_scenario,
                         engines={"A":"tree","B":"bbn","C":"bbn"},
                         projection="aware", thd=0.85)
    commitment_breakdown(spec, obs, hard_scenario,
                         engines={"A":"tree","B":"bbn","C":"bbn"},
                         projection="naive", thd=0.85)


if __name__ == "__main__":
    main()

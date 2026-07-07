"""
run_n5_instrument.py
====================
Team-size scalability check: N=5 EMBM agents on Domain17, with the same
conflict-count instrumentation as run_conflict_instrument.py.

Design (locked with Ravi 2026-07-06):
  * Domain17 network (same as N=3 primary experiments)
  * 5-agent observability partition (make_observability17_n5) — 2 obs/agent,
    2 overlaps, 8 unique observed + 9 hidden inference targets
  * Outlier composition: Agent A is the mismatched substrate; B/C/D/E all BBN
  * 2x2 design: {matched, mismatched} x {aware, naive}, on tree and markov axes
  * N=5 replicates per (scenario, cell) — matched to the ALARM scale-expansion
    budget, defensible as a scalability check
  * Top 3 hard scenarios from the existing Domain17 pool (pool_seed 20260706)
  * Base seeds: tree = 1_100_000, markov = 1_110_000

EMBM agents already support arbitrary team size N via `teammates` list;
belief structures per agent = 1 + (N-1) + (N-1)^2 = 21 for N=5 (vs 7 for N=3).
No refactor to EMBMAgent, build_team, _resolve, or find_conflicts was needed.
"""
from __future__ import annotations
import json
import os
import pickle
import sys
import time
from collections import Counter
import numpy as np
import random

from bbn_engine import UNCERTAIN
from experiment import (
    TeamConfig, build_team, _resolve, team_accuracy
)
from simulation import node_importance


THD = 0.55
N_RUNS = int(os.environ.get("N_RUNS", 5))
CONFLICT_TYPES = ("I", "II", "III")
AGENT_NAMES = ["A", "B", "C", "D", "E"]
OUTLIER = "A"   # convention: first agent is the substrate outlier


def snapshot_conflicts(agents, scenario):
    """Passive snapshot of every conflict currently perceived by every agent.
    Same instrumentation as run_conflict_instrument.py, but the by_agent
    breakdown iterates over the actual team roster (N-generic)."""
    total = 0
    by_type = Counter()
    by_agent = Counter()
    phantom = 0
    real = 0
    for ag in agents:
        confs = ag.find_conflicts()
        for (node, ctype, mate) in confs:
            total += 1
            by_type[ctype] += 1
            by_agent[ag.name] += 1
            self_diag = ag.M.diagnosis(node, ag.thd)
            truth = scenario.truth[node]
            if self_diag == truth:
                phantom += 1
            else:
                real += 1
    return {
        "total": total,
        "by_type": {t: by_type.get(t, 0) for t in CONFLICT_TYPES},
        "by_agent": {ag.name: by_agent.get(ag.name, 0) for ag in agents},
        "phantom": phantom,
        "real": real,
    }


def run_once_instrumented(spec, observability, scenario, tc, rng, importance):
    """RNG-faithful mirror of experiment.run_once with conflict-count telemetry."""
    agents = build_team(spec, observability, tc)
    by_name = {a.name: a for a in agents}
    curve = [team_accuracy(agents, scenario)]

    # observation phase
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
        curve.append(team_accuracy(agents, scenario))

    init = snapshot_conflicts(agents, scenario)

    # cooperation phase
    def diag_state(ags):
        return tuple(tuple(a.M.diagnosis(n, tc.thd) for n in spec.nodes) for a in ags)

    idle = 0
    last = diag_state(agents)
    while len(curve) - 1 < tc.max_actions:
        if all(not a.find_conflicts() for a in agents):
            break
        actor = rng.choice(agents)
        conflicts = actor.find_conflicts()
        if not conflicts:
            continue
        node, ctype, mate = max(conflicts, key=lambda c: importance[c[0]])
        _resolve(actor, by_name, importance, node, ctype, mate, scenario, tc)
        curve.append(team_accuracy(agents, scenario))
        now = diag_state(agents)
        if now == last:
            idle += 1
            if idle >= tc.idle_limit:
                break
        else:
            idle = 0
            last = now

    final = snapshot_conflicts(agents, scenario)

    return {
        "final_acc": curve[-1],
        "actions": len(curve) - 1,
        "initial": init,
        "final_conflicts": final,
    }


def run_many_instrumented(spec, obs, scenario, tc, n_runs, seed):
    """Aggregate across runs. Same output shape as N=3 runner but by_agent
    dict has 5 entries."""
    rng = random.Random(seed)
    importance = node_importance(spec)
    finals, actions = [], []
    init_totals, init_phantoms, init_reals = [], [], []
    init_by_type = {t: [] for t in CONFLICT_TYPES}
    init_by_agent = {n: [] for n in AGENT_NAMES}
    final_totals = []
    for _ in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        r = run_once_instrumented(spec, obs, scenario, tc, sub, importance)
        finals.append(r["final_acc"])
        actions.append(r["actions"])
        init_totals.append(r["initial"]["total"])
        init_phantoms.append(r["initial"]["phantom"])
        init_reals.append(r["initial"]["real"])
        for t in CONFLICT_TYPES:
            init_by_type[t].append(r["initial"]["by_type"][t])
        for n in AGENT_NAMES:
            init_by_agent[n].append(r["initial"]["by_agent"][n])
        final_totals.append(r["final_conflicts"]["total"])
    return {
        "final_mean": float(np.mean(finals)),
        "final_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "actions_mean": float(np.mean(actions)),
        "init_conflicts_mean": float(np.mean(init_totals)),
        "init_conflicts_sd": float(np.std(init_totals, ddof=1)) if len(init_totals) > 1 else 0.0,
        "init_phantom_mean": float(np.mean(init_phantoms)),
        "init_real_mean": float(np.mean(init_reals)),
        "init_by_type_mean": {t: float(np.mean(init_by_type[t])) for t in CONFLICT_TYPES},
        "init_by_agent_mean": {n: float(np.mean(init_by_agent[n])) for n in AGENT_NAMES},
        "final_conflicts_mean": float(np.mean(final_totals)),
    }


def make_cells(axis):
    """Build the 2x2 cells for N=5 with Agent A as the substrate outlier."""
    matched = {nm: "bbn" for nm in AGENT_NAMES}
    mismatched = {nm: "bbn" for nm in AGENT_NAMES}
    mismatched[OUTLIER] = axis
    return {
        "matched-aware":    dict(engines=matched,    projection="aware"),
        "matched-naive":    dict(engines=matched,    projection="naive"),
        "mismatched-aware": dict(engines=mismatched, projection="aware"),
        "mismatched-naive": dict(engines=mismatched, projection="naive"),
    }


def run_2x2_instrumented(spec, obs, scenario, axis, base_seed, seed_offset):
    cells = make_cells(axis)
    out = {}
    for i, (name, cfg) in enumerate(cells.items()):
        tc = TeamConfig(engines=cfg["engines"], projection_mode=cfg["projection"],
                        charitable=False, thd=THD, gamma=0.95,
                        max_actions=100, idle_limit=5)
        out[name] = run_many_instrumented(spec, obs, scenario, tc,
                                          n_runs=N_RUNS,
                                          seed=base_seed + seed_offset + i)
    return out


def load_config(network, axis):
    """Return (spec, obs, top-3 scenarios, base_seed) for the given (network, axis).

    Base seeds:
      domain17 tree   = 1_100_000, markov = 1_110_000
      kanno    tree   = 1_200_000, markov = 1_210_000
      alarm    tree   = 1_300_000, markov = 1_310_000
    """
    if network == "domain17":
        from domain17 import make_domain17, make_observability17_n5
        spec = make_domain17()
        obs = make_observability17_n5()
        pool_path_candidates = [
            "./results/hard_scenarios_domain17.pkl",
            "/home/claude/substrate_abm/hard_scenarios_domain17.pkl",
            "hard_scenarios_domain17.pkl",
        ]
        base_seed = {"tree": 1_100_000, "markov": 1_110_000}[axis]
    elif network == "kanno":
        from domain_real import make_spec_shared, make_observability_real_n5
        spec = make_spec_shared("agent1")
        obs = make_observability_real_n5()
        pool_path_candidates = [
            "./results/hard_scenarios.pkl",
            "/home/claude/substrate_abm/hard_scenarios.pkl",
            "hard_scenarios.pkl",
        ]
        base_seed = {"tree": 1_200_000, "markov": 1_210_000}[axis]
    elif network == "alarm":
        from alarm_domain import make_alarm, make_observability_alarm_n5
        spec = make_alarm()
        obs = make_observability_alarm_n5()
        pool_path_candidates = [
            "./results/hard_scenarios_alarm.pkl",
            "hard_scenarios_alarm.pkl",
        ]
        base_seed = {"tree": 1_300_000, "markov": 1_310_000}[axis]
    else:
        raise ValueError(f"unknown network: {network}")

    for p in pool_path_candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                scenarios = pickle.load(f)["scenarios"]
            break
    else:
        raise FileNotFoundError(
            f"Could not locate hard-scenarios pool. Tried: {pool_path_candidates}")
    scenarios = scenarios[:3]  # top 3
    return spec, obs, scenarios, base_seed


def main():
    if (len(sys.argv) != 3
            or sys.argv[1] not in ("domain17", "kanno", "alarm")
            or sys.argv[2] not in ("tree", "markov")):
        print("Usage: python run_n5_instrument.py {domain17|kanno|alarm} {tree|markov}")
        sys.exit(1)
    network, axis = sys.argv[1], sys.argv[2]

    spec, obs, scenarios, base_seed = load_config(network, axis)

    out_path = f"./results/n5_instrument_{network}_{axis}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {"axis": axis, "network": network, "thd": THD, "n_runs": N_RUNS,
                "team_size": 5, "outlier": OUTLIER, "results": []}
    done = {r["rank"] for r in data["results"]}

    print(f"N=5 team-size {network} {axis}-axis 2x2, "
          f"THD={THD}, N_replicates={N_RUNS}, {len(scenarios)} scenarios\n", flush=True)

    for rank, (idx, sc, h) in enumerate(scenarios, start=1):
        if rank in done:
            print(f"  [skip] #{rank}", flush=True)
            continue
        ts = time.time()
        cells = run_2x2_instrumented(spec, obs, sc, axis, base_seed, rank*100)
        dt = time.time() - ts
        rec = {"rank": rank, "pool_idx": idx, "hardness": h, "seconds": dt}
        for name, c in cells.items():
            for k, v in c.items():
                rec[f"{name}__{k}"] = v
        data["results"].append(rec)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        ma = cells["matched-aware"]; mia = cells["mismatched-aware"]
        min_ = cells["mismatched-naive"]
        print(f"  #{rank} h={h:2d}: "
              f"init_conf m-a={ma['init_conflicts_mean']:5.1f} "
              f"mm-a={mia['init_conflicts_mean']:5.1f} "
              f"mm-n={min_['init_conflicts_mean']:5.1f}  |  "
              f"acc m-a={ma['final_mean']:.3f} "
              f"mm-a={mia['final_mean']:.3f} "
              f"mm-n={min_['final_mean']:.3f}  [{dt:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

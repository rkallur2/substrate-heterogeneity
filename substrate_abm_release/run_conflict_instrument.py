"""
run_conflict_instrument.py
==========================
Domain17 Markov 2x2 with CONFLICT-COUNT instrumentation.

For each run we record, in addition to accuracy:
  - initial_conflicts:   total conflicts across the team right after obs phase,
                         before any cooperation actions -- the "detection
                         sensitivity" measure. Broken down by type (I/II/III)
                         and by agent (A/B/C).
  - final_conflicts:     total conflicts across the team at end of the run --
                         residual disagreement.
  - phantom_initial:     of the initial conflicts, how many involved an actor
                         whose own diagnosis DISAGREED with ground truth
                         (i.e., the conflict was 'real' from the environment's
                         POV -- the actor was right to feel uncertain).
                         The complement (real_initial) is initial - phantom.
                         Convention:
                           conflict IS phantom  ==> actor's L1 diagnosis == truth
                                                    (their disagreement with the
                                                    teammate model was illusory
                                                    from an oracle's POV)
                           conflict IS real     ==> actor's L1 diagnosis != truth
                                                    (they were genuinely wrong
                                                    and needed the teammate)
  - actions:             already tracked -- cooperation actions taken.

The purpose: test whether the mismatched-aware advantage on domain17 Markov
comes from detecting MORE conflicts (so richer coordination) or FEWER
conflicts (so avoiding costly wrong actions). Same 8 hard scenarios, same
RNG seed convention as run_domain17_2x2.py (base_seed=910_000 for Markov).

Reproducibility: this file's run_once_instrumented preserves the RNG stream
of experiment.run_once (same rng.shuffle order, same rng.choice sequence).
Instrumentation is passive -- it inspects agent state without mutating it.
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
N_RUNS = 15   # matched to Option 1 (dual-metric) run for direct comparability
CONFLICT_TYPES = ("I", "II", "III")  # match embm_agent.TYPE_I/II/III string values


def snapshot_conflicts(agents, scenario):
    """Take a passive snapshot of every conflict currently perceived by every
    agent. Returns dict with total, by_type, by_agent, and phantom counts.
    A conflict (node, ctype, mate) is 'phantom' if actor.M.diagnosis(node) ==
    scenario.truth[node] (i.e., the actor is already right; the perceived
    mismatch with the teammate model would waste an action)."""
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
    """RNG-faithful mirror of experiment.run_once with conflict-count telemetry.
    Every RNG call is placed in the same order as run_once so the accuracy
    result is identical at matched seeds."""
    agents = build_team(spec, observability, tc)
    by_name = {a.name: a for a in agents}
    curve = [team_accuracy(agents, scenario)]

    # observation phase (rng.shuffle used here, matches run_once)
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
        curve.append(team_accuracy(agents, scenario))

    # INITIAL conflict snapshot (after obs, before cooperation)
    init = snapshot_conflicts(agents, scenario)

    # cooperation phase (rng.choice used here, matches run_once)
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

    # FINAL conflict snapshot
    final = snapshot_conflicts(agents, scenario)

    return {
        "final_acc": curve[-1],
        "actions": len(curve) - 1,
        "initial": init,
        "final_conflicts": final,
    }


def run_many_instrumented(spec, obs, scenario, tc, n_runs, seed):
    """Aggregate across runs. Returns cell-level means (and per-run arrays for
    stats downstream)."""
    rng = random.Random(seed)
    importance = node_importance(spec)
    finals, actions = [], []
    init_totals, init_phantoms, init_reals = [], [], []
    init_by_type = {t: [] for t in CONFLICT_TYPES}
    init_by_agent = {n: [] for n in ("A", "B", "C")}
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
        for n in ("A", "B", "C"):
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
        "init_by_agent_mean": {n: float(np.mean(init_by_agent[n])) for n in ("A", "B", "C")},
        "final_conflicts_mean": float(np.mean(final_totals)),
    }


def run_2x2_instrumented(spec, obs, scenario, axis, base_seed, seed_offset):
    cells = {
        "matched-aware":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="aware"),
        "matched-naive":    dict(engines={"A":"bbn","B":"bbn","C":"bbn"},   projection="naive"),
        "mismatched-aware": dict(engines={"A":axis,"B":"bbn","C":"bbn"},    projection="aware"),
        "mismatched-naive": dict(engines={"A":axis,"B":"bbn","C":"bbn"},    projection="naive"),
    }
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
    """Return (spec, obs, scenarios, base_seed) for the given (network, axis).
    Base seeds match the ORIGINAL harness scripts so accuracy is directly
    comparable to prior JSONs:
        domain17 tree   -> run_domain17_2x2.py used 900_000
        domain17 markov -> run_domain17_2x2.py used 910_000
        kanno tree      -> run_multiscen_2x2.py used 500_000
        kanno markov    -> run_markov_2x2.py     (kept for later)
    """
    if network == "domain17":
        from domain17 import make_domain17, make_observability17
        spec = make_domain17()
        obs = make_observability17()
        pool_path = "./results/hard_scenarios_domain17.pkl"
        base_seed = {"tree": 900_000, "markov": 910_000}[axis]
    elif network == "kanno":
        from domain_real import make_spec_shared, make_observability_real
        spec = make_spec_shared("agent1")
        obs = make_observability_real()
        pool_path = "./results/hard_scenarios.pkl"
        base_seed = {"tree": 500_000, "markov": 800_000}[axis]
    else:
        raise ValueError(f"unknown network: {network}")
    with open(pool_path, "rb") as f:
        scenarios = pickle.load(f)["scenarios"]
    return spec, obs, scenarios, base_seed


def main():
    if (len(sys.argv) != 3
            or sys.argv[1] not in ("domain17", "kanno")
            or sys.argv[2] not in ("tree", "markov")):
        print("Usage: python run_conflict_instrument.py {domain17|kanno} {tree|markov}")
        sys.exit(1)
    network, axis = sys.argv[1], sys.argv[2]

    spec, obs, scenarios, base_seed = load_config(network, axis)

    out_path = f"./results/conflict_instrument_{network}_{axis}.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {"axis": axis, "thd": THD, "n_runs": N_RUNS, "results": []}
    done = {r["rank"] for r in data["results"]}

    print(f"Conflict-instrument {network} {axis}-axis 2x2, "
          f"THD={THD}, N={N_RUNS}, {len(scenarios)} scenarios\n", flush=True)

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
        # concise progress print
        ma = cells["matched-aware"]; mia = cells["mismatched-aware"]; min_ = cells["mismatched-naive"]
        print(f"  #{rank} h={h:2d}: "
              f"init_conf m-a={ma['init_conflicts_mean']:5.1f} "
              f"mm-a={mia['init_conflicts_mean']:5.1f} "
              f"mm-n={min_['init_conflicts_mean']:5.1f}  |  "
              f"acc m-a={ma['final_mean']:.3f} mm-a={mia['final_mean']:.3f} "
              f"mm-n={min_['final_mean']:.3f}  [{dt:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

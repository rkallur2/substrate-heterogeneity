"""
run_dual_metric.py
==================
Substrate 2x2 with DUAL METRIC reporting:
  - full accuracy: correct / N  (UNCERTAIN counted as wrong -- the metric used
    throughout the study, matching Okabe/Kanno's team_accuracy)
  - committed accuracy: correct / #committed  (drops UNCERTAIN from denominator)

The test: does the domain17 Markov > BBN result reverse under the committed
metric? If yes, the "Markov's advantage is commitment bias" mechanism is
confirmed and Markov's substrate gap sign is confounded with abstention behavior.
Also runs the other three cells (Kanno tree/Markov, domain17 tree) as sanity
checks.

Usage:
    python run_dual_metric.py kanno tree
    python run_dual_metric.py kanno markov
    python run_dual_metric.py domain17 tree
    python run_dual_metric.py domain17 markov
"""
from __future__ import annotations
import json
import os
import pickle
import random
import sys
import time
import numpy as np

from bbn_engine import UNCERTAIN
from experiment import (
    TeamConfig, build_team, _resolve, engine_kind, FACTORIES,
)
from simulation import node_importance


THD = 0.55
N_RUNS = 15   # reduced from 40 to fit within container's per-call time budget


def dual_team_accuracy(agents, scenario):
    """Return (full_acc, committed_acc, mean_commit_rate):
       full: mean per-agent (correct / N)  -- UNCERTAIN as wrong
       committed: mean per-agent (correct / #committed)  -- drop UNCERTAIN
       commit_rate: mean per-agent (#committed / N)
    """
    N = len(agents[0].spec.nodes)
    full_accs = []
    committed_accs = []
    commit_rates = []
    for ag in agents:
        correct = 0
        committed = 0
        for node, truth in scenario.truth.items():
            d = ag.M.diagnosis(node, ag.thd)
            if d == UNCERTAIN:
                continue
            committed += 1
            if d == truth:
                correct += 1
        full_accs.append(correct / N)
        if committed > 0:
            committed_accs.append(correct / committed)
        # else: skip this agent for committed-accuracy averaging (undefined)
        commit_rates.append(committed / N)
    full_mean = float(np.mean(full_accs))
    committed_mean = (float(np.mean(committed_accs)) if committed_accs
                      else float("nan"))
    commit_rate_mean = float(np.mean(commit_rates))
    return full_mean, committed_mean, commit_rate_mean


def run_once_dual(spec, observability, scenario, tc, rng, importance, agents=None):
    """Run one simulation, return dual-metric result at end-of-run."""
    if agents is None:
        agents = build_team(spec, observability, tc)
    by_name = {a.name: a for a in agents}

    # observation phase
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])

    # cooperation phase (mirrors experiment.run_once)
    def diag_state(ags):
        return tuple(tuple(a.M.diagnosis(n, tc.thd) for n in spec.nodes) for a in ags)

    idle = 0
    last = diag_state(agents)
    actions = 0
    while actions < tc.max_actions:
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

    full, committed, commit_rate = dual_team_accuracy(agents, scenario)
    return {
        "full": full, "committed": committed, "commit_rate": commit_rate,
        "actions": actions,
    }


def run_many_dual(spec, obs, scenario, tc, n_runs, seed):
    rng = random.Random(seed)
    importance = node_importance(spec)
    full_vals, comm_vals, cr_vals, act_vals = [], [], [], []
    for _ in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        r = run_once_dual(spec, obs, scenario, tc, sub, importance)
        full_vals.append(r["full"])
        if not np.isnan(r["committed"]):
            comm_vals.append(r["committed"])
        cr_vals.append(r["commit_rate"])
        act_vals.append(r["actions"])
    return {
        "full_mean": float(np.mean(full_vals)),
        "full_sd": float(np.std(full_vals)),
        "committed_mean": (float(np.mean(comm_vals)) if comm_vals else float("nan")),
        "committed_sd": (float(np.std(comm_vals)) if comm_vals else float("nan")),
        "commit_rate_mean": float(np.mean(cr_vals)),
        "actions_mean": float(np.mean(act_vals)),
    }


def load_config(network, axis):
    if network == "kanno":
        from domain_real import make_spec_shared, make_observability_real
        spec = make_spec_shared("agent1")
        obs = make_observability_real()
        pool_path = "./results/hard_scenarios.pkl"
    elif network == "domain17":
        from domain17 import make_domain17, make_observability17
        spec = make_domain17()
        obs = make_observability17()
        pool_path = "./results/hard_scenarios_domain17.pkl"
    else:
        raise ValueError(network)
    with open(pool_path, "rb") as f:
        scenarios = pickle.load(f)["scenarios"]
    return spec, obs, scenarios


def run_2x2_dual(spec, obs, scenario, axis, base_seed, seed_offset):
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
        out[name] = run_many_dual(spec, obs, scenario, tc, n_runs=N_RUNS,
                                  seed=base_seed + seed_offset + i)
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_dual_metric.py {kanno|domain17} {tree|markov}")
        sys.exit(1)
    network, axis = sys.argv[1], sys.argv[2]
    if network not in ("kanno", "domain17") or axis not in ("tree", "markov"):
        print("Bad args")
        sys.exit(1)

    out_path = f"./results/dual_metric_{network}_{axis}.json"
    # Match base_seeds from the original single-metric runs so FULL numbers
    # can be checked against the prior JSONs (run_domain17_2x2.py used 900_000
    # / 910_000 for domain17 tree / markov).
    base_seed = {("kanno","tree"):700_000, ("kanno","markov"):800_000,
                 ("domain17","tree"):900_000, ("domain17","markov"):910_000}[(network, axis)]

    spec, obs, scenarios = load_config(network, axis)
    print(f"Dual-metric {network} {axis}-axis 2x2, THD={THD}, N={N_RUNS}\n")

    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {"network": network, "axis": axis, "thd": THD, "n_runs": N_RUNS,
                "results": []}
    done = {r["rank"] for r in data["results"]}

    t_start = time.time()
    for rank, (idx, sc, h) in enumerate(scenarios, start=1):
        if rank in done:
            print(f"  [skip] #{rank}")
            continue
        ts = time.time()
        cells = run_2x2_dual(spec, obs, sc, axis=axis,
                             base_seed=base_seed, seed_offset=rank*100)
        dt = time.time() - ts
        # extract metrics
        ma_f  = cells["matched-aware"]["full_mean"];   ma_c  = cells["matched-aware"]["committed_mean"];   ma_cr = cells["matched-aware"]["commit_rate_mean"]
        mia_f = cells["mismatched-aware"]["full_mean"]; mia_c = cells["mismatched-aware"]["committed_mean"]; mia_cr = cells["mismatched-aware"]["commit_rate_mean"]
        min_f = cells["mismatched-naive"]["full_mean"]; min_c = cells["mismatched-naive"]["committed_mean"]; min_cr = cells["mismatched-naive"]["commit_rate_mean"]
        sub_gap_full = ma_f - mia_f
        sub_gap_committed = ma_c - mia_c
        aware_gap_full = mia_f - min_f
        aware_gap_committed = mia_c - min_c
        record = {
            "rank": rank, "pool_idx": idx, "hardness": h,
            "matched_aware_full": ma_f, "matched_aware_committed": ma_c,
            "matched_aware_commit_rate": ma_cr,
            "mismatched_aware_full": mia_f, "mismatched_aware_committed": mia_c,
            "mismatched_aware_commit_rate": mia_cr,
            "mismatched_naive_full": min_f, "mismatched_naive_committed": min_c,
            "mismatched_naive_commit_rate": min_cr,
            "sub_gap_full": sub_gap_full, "sub_gap_committed": sub_gap_committed,
            "aware_gap_full": aware_gap_full, "aware_gap_committed": aware_gap_committed,
            "seconds": dt,
        }
        data["results"].append(record)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  #{rank} h={h:2d}: "
              f"FULL sub={sub_gap_full:+.3f} aware={aware_gap_full:+.3f}  |  "
              f"COMM sub={sub_gap_committed:+.3f} aware={aware_gap_committed:+.3f}  |  "
              f"commit_rate m-a={ma_cr:.2f} mm-a={mia_cr:.2f} mm-n={min_cr:.2f}  "
              f"[{dt:.0f}s]")

    R = data["results"]
    if not R:
        return
    print("\n" + "=" * 100)
    print(f"AGGREGATE  {network} {axis}-axis  (N={len(R)} scenarios, THD={THD})")
    print("=" * 100)
    def _mean(k): return float(np.mean([r[k] for r in R]))
    print(f"  matched-aware   :  full={_mean('matched_aware_full'):.3f}   "
          f"committed={_mean('matched_aware_committed'):.3f}   "
          f"commit_rate={_mean('matched_aware_commit_rate'):.3f}")
    print(f"  mismatched-aware:  full={_mean('mismatched_aware_full'):.3f}   "
          f"committed={_mean('mismatched_aware_committed'):.3f}   "
          f"commit_rate={_mean('mismatched_aware_commit_rate'):.3f}")
    print(f"  mismatched-naive:  full={_mean('mismatched_naive_full'):.3f}   "
          f"committed={_mean('mismatched_naive_committed'):.3f}   "
          f"commit_rate={_mean('mismatched_naive_commit_rate'):.3f}")
    print(f"\n  sub_gap  FULL      = {_mean('sub_gap_full'):+.3f}   "
          f"sub_gap COMMITTED = {_mean('sub_gap_committed'):+.3f}")
    print(f"  aware_gap FULL     = {_mean('aware_gap_full'):+.3f}   "
          f"aware_gap COMMITTED = {_mean('aware_gap_committed'):+.3f}")


if __name__ == "__main__":
    main()

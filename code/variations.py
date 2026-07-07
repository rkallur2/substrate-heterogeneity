"""
variations.py
=============
The five model variations from Okabe/Kanno Table 4, used as the Phase-1
*validation target*: a faithful reimplementation should reproduce their
qualitative findings --

  * MB models reach a given accuracy in FEWER actions than non-MB (efficiency)
  * MB models have LOWER variance across runs than non-MB (stability)

Variations:
  non-MB0 : observation only; no communication (L1 only, no speaking)
  non-MB1 : observation + broadcast results; L1 only, but states are shared
  MB0     : L1 + L2 (Type I conflicts only)
  MB1     : L1 + L3 (Types II, III)
  MB2     : L1 + L2 + L3 (all conflict types)   <- the full model tested above

These are configured by (a) which conflict types an agent will act on, and
(b) whether communication happens at all.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set
import random
import numpy as np

from bbn_engine import BBNEngine
from simulation import (SimConfig, build_team, node_importance, team_accuracy,
                        Scenario, _resolve)
from embm_agent import TYPE_I, TYPE_II, TYPE_III


VARIATION_CONFLICTS = {
    "non-MB0": set(),                       # never resolves conflicts (no comm)
    "non-MB1": set(),                       # broadcasts observations only
    "MB0":     {TYPE_I},
    "MB1":     {TYPE_II, TYPE_III},
    "MB2":     {TYPE_I, TYPE_II, TYPE_III},
}
NON_COMMUNICATING = {"non-MB0"}             # observes only, never hears others
BROADCAST_ONLY = {"non-MB1"}                # shares observations, no belief reasoning


def run_variation(spec, observability, scenario: Scenario, cfg: SimConfig,
                  variation: str, rng: random.Random):
    """Run one simulation under a given model variation. Returns accuracy curve."""
    allowed = VARIATION_CONFLICTS[variation]
    agents = build_team(spec, observability, cfg)
    by_name = {a.name: a for a in agents}
    importance = node_importance(spec)
    curve = [team_accuracy(agents, scenario)]

    # ---- observation phase ----
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
        if variation in BROADCAST_ONLY:
            # non-MB1: the observation is broadcast; teammates fold it into L1 directly
            for other in agents:
                if other is not ag:
                    other.M.ingest(node, other._soft_dist(node, scenario.truth[node]))
        curve.append(team_accuracy(agents, scenario))

    # ---- cooperation phase ----
    if variation in NON_COMMUNICATING or not allowed:
        return curve  # non-MB0 stops after observation; non-MB1 has no belief-reasoning

    def diag_state(ags):
        return tuple(tuple(a.M.diagnosis(n, cfg.thd) for n in spec.nodes) for a in ags)

    idle = 0
    last = diag_state(agents)
    while len(curve) - 1 < cfg.max_actions:
        # consider only conflicts of allowed types
        if all(not [c for c in a.find_conflicts() if c[1] in allowed] for a in agents):
            break
        actor = rng.choice(agents)
        conflicts = [c for c in actor.find_conflicts() if c[1] in allowed]
        if not conflicts:
            continue
        node, ctype, mate = max(conflicts, key=lambda c: importance[c[0]])
        _resolve(actor, by_name, node, ctype, mate, scenario, cfg)
        curve.append(team_accuracy(agents, scenario))
        now = diag_state(agents)
        if now == last:
            idle += 1
            if idle >= cfg.idle_limit:
                break
        else:
            idle = 0
            last = now
    return curve


def run_variation_many(spec, observability, scenario, cfg, variation,
                       n_runs=100, seed=0):
    rng = random.Random(seed)
    curves, finals, lengths = [], [], []
    for _ in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        c = run_variation(spec, observability, scenario, cfg, variation, sub)
        curves.append(c); finals.append(c[-1]); lengths.append(len(c) - 1)
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
        arr[i, len(c):] = c[-1]
    return {
        "variation": variation,
        "mean_curve": arr.mean(axis=0),
        "sd_curve": arr.std(axis=0),
        "final_mean": float(np.mean(finals)),
        "final_sd": float(np.std(finals)),
        "actions_mean": float(np.mean(lengths)),
        "actions_sd": float(np.std(lengths)),
    }

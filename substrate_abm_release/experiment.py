"""
experiment.py
=============
Phase 4-6: the substrate-heterogeneity experiment harness.

Builds teams with configurable engine composition and projection mode, runs the
two-phase simulation with TRANSLATION engaged on every communication, and returns
the Okabe/Kanno dependent variables plus the new ones (Translational Friction,
illusory-agreement onset).

Composition (decided): mismatched teams = ONE tree agent + TWO probabilistic
(BBN) agents, isolating the false-confidence injector. Matched teams = all-BBN
(the Okabe/Kanno baseline) or all-Markov / all-tree homogeneous baselines.

The 2x2 manipulation:
                    matched substrate        mismatched substrate
   aware M''        control (= Phase 1)      gap, modelled correctly
   naive M''        gap absent, harmless     Anthropomorphization Trap
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
import random
import numpy as np

from bbn_engine import CausalSpec, BBNEngine, InferenceEngine, UNCERTAIN
from markov_engine import MarkovEngine
from tree_engine import DecisionTreeEngine
from embm_agent import EMBMAgent, _proj_key, TYPE_I, TYPE_II, TYPE_III
from simulation import node_importance, team_accuracy, Scenario, sample_scenario
from translation import make_message, translate, KIND_PROB, KIND_TREE


# engine kind tag from a factory/instance
def engine_kind(factory_or_obj) -> str:
    name = (factory_or_obj.__name__ if isinstance(factory_or_obj, type)
            else factory_or_obj.__class__.__name__)
    return KIND_TREE if name == "DecisionTreeEngine" else KIND_PROB


FACTORIES = {"bbn": BBNEngine, "markov": MarkovEngine, "tree": DecisionTreeEngine}


@dataclass
class TeamConfig:
    # map agent name -> engine key ("bbn"/"markov"/"tree")
    engines: Dict[str, str]
    projection_mode: str = "aware"     # "aware" | "naive"
    charitable: bool = False
    thd: float = 0.75
    gamma: float = 0.95
    max_actions: int = 100
    idle_limit: int = 5


def build_team(spec: CausalSpec, observability: Dict[str, List[str]],
               tc: TeamConfig) -> List[EMBMAgent]:
    names = list(observability.keys())
    factories = {nm: FACTORIES[tc.engines[nm]] for nm in names}
    agents = []
    for nm in names:
        mates = [m for m in names if m != nm]
        agents.append(EMBMAgent(
            name=nm, teammates=mates, spec=spec,
            engine_factory=factories[nm],
            observable=observability[nm], thd=tc.thd, gamma=tc.gamma,
            teammate_factories={m: factories[m] for m in mates},  # TRUE substrates
            projection_mode=tc.projection_mode,
        ))
    return agents


def build_team_parametric(specs: Dict[str, CausalSpec],
                          observability: Dict[str, List[str]],
                          tc: TeamConfig) -> List[EMBMAgent]:
    """PARAMETRIC-HETEROGENEITY team: each agent runs its OWN spec (its own CPTs),
    same structure. This is a SEPARATE manipulation from substrate heterogeneity.

    Design (tractable-realistic, applied to parametric case): agent A's entire
    belief apparatus -- its L1 self-cognition AND its models of teammates -- runs
    on A's OWN CPTs, because A has no access to teammates' private probability
    estimates. The belief mismatch emerges naturally: A's model of B (built with
    A's CPTs) diverges from B's actual beliefs (B's CPTs). This is parametric loss,
    kept distinct from substrate (engine-type) loss.

    All specs MUST share node structure (validated by caller). Engines are still
    set by tc.engines so parametric and substrate axes can be studied separately
    OR crossed (with the caveat that crossing them confounds the two -- prefer
    holding engines uniform when isolating the parametric effect).
    """
    names = list(observability.keys())
    factories = {nm: FACTORIES[tc.engines[nm]] for nm in names}
    agents = []
    for nm in names:
        mates = [m for m in names if m != nm]
        agents.append(EMBMAgent(
            name=nm, teammates=mates, spec=specs[nm],   # <-- agent's OWN spec/CPTs
            engine_factory=factories[nm],
            observable=observability[nm], thd=tc.thd, gamma=tc.gamma,
            teammate_factories={m: factories[m] for m in mates},
            projection_mode=tc.projection_mode,
        ))
    return agents


def _resolve(actor: EMBMAgent, by_name: Dict[str, EMBMAgent], importance,
             node: str, ctype: str, mate: str, scenario: Scenario, tc: TeamConfig):
    """Conflict resolution with TRANSLATION engaged on every communication."""
    self_diag = actor.M.diagnosis(node, actor.thd)
    teammate = by_name[mate]
    actor_kind = engine_kind(actor.M)
    mate_kind = engine_kind(teammate.M)

    if self_diag == UNCERTAIN:
        # QUERY: ask the teammate; they answer with their diagnosis (their message)
        msg = make_message(teammate.M, mate_kind, node, teammate.thd)
        # actor hears the answer (translate mate -> actor's divisions)
        actor.hear_inform(mate, node, msg, sender_kind=mate_kind,
                          charitable=tc.charitable, sender_engine=teammate.M)
        # third agent overhears the answer
        for other in actor.teammates:
            if other != mate:
                by_name[other].hear_inform(mate, node, msg, sender_kind=mate_kind,
                                           charitable=tc.charitable, sender_engine=teammate.M)
    elif node in actor.observable:
        # LOOK-AT: settle directly (no translation; own observation)
        actor.observe(node, scenario.truth[node])
    else:
        # INFORM: actor pushes its diagnosis to the teammate (+ third overhears)
        msg = make_message(actor.M, actor_kind, node, actor.thd)
        teammate.hear_inform(actor.name, node, msg, sender_kind=actor_kind,
                             charitable=tc.charitable, sender_engine=actor.M)
        for other in actor.teammates:
            if other != mate:
                by_name[other].hear_inform(actor.name, node, msg, sender_kind=actor_kind,
                                           charitable=tc.charitable, sender_engine=actor.M)


def run_once(spec, observability, scenario, tc: TeamConfig, rng: random.Random,
             importance=None, agents=None):
    """Run one simulation. If `agents` is provided (e.g. from build_team_parametric)
    it is used as-is; otherwise a homogeneous-spec team is built via build_team.
    `spec` is still used for structure-level quantities (importance, node list,
    accuracy scoring) -- valid because all parametric specs share structure."""
    if agents is None:
        agents = build_team(spec, observability, tc)
    by_name = {a.name: a for a in agents}
    if importance is None:
        importance = node_importance(spec)
    curve = [team_accuracy(agents, scenario)]

    # observation phase
    obs_actions = [(a, n) for a in agents for n in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
        curve.append(team_accuracy(agents, scenario))

    # cooperation phase
    def diag_state(ags):
        return tuple(tuple(a.M.diagnosis(n, tc.thd) for n in spec.nodes) for a in ags)

    idle = 0
    last = diag_state(agents)
    suppressed_at = None  # illusory-agreement onset: first action where an agent
                          # had a conflict but believed (via L3) alignment existed
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

    return {"curve": curve, "actions": len(curve) - 1, "final": curve[-1]}


def run_many(spec, observability, scenario, tc: TeamConfig, n_runs=100, seed=0):
    rng = random.Random(seed)
    importance = node_importance(spec)   # invariant across runs -- compute once
    curves, finals, lengths = [], [], []
    for _ in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        r = run_once(spec, observability, scenario, tc, sub, importance=importance)
        curves.append(r["curve"]); finals.append(r["final"]); lengths.append(r["actions"])
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c; arr[i, len(c):] = c[-1]
    return {
        "mean_curve": arr.mean(axis=0), "sd_curve": arr.std(axis=0),
        "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals)),
        "actions_mean": float(np.mean(lengths)), "actions_sd": float(np.std(lengths)),
    }


def run_many_parametric(specs: Dict[str, CausalSpec], observability, scenario,
                        tc: TeamConfig, n_runs=100, seed=0, struct_spec=None):
    """PARAMETRIC-heterogeneity Monte Carlo: each agent runs its own spec's CPTs.
    `struct_spec` supplies the shared structure for importance/accuracy/node list
    (defaults to the first agent's spec, since all share structure)."""
    if struct_spec is None:
        struct_spec = next(iter(specs.values()))
    rng = random.Random(seed)
    importance = node_importance(struct_spec)
    curves, finals, lengths = [], [], []
    for _ in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        agents = build_team_parametric(specs, observability, tc)
        r = run_once(struct_spec, observability, scenario, tc, sub,
                     importance=importance, agents=agents)
        curves.append(r["curve"]); finals.append(r["final"]); lengths.append(r["actions"])
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c; arr[i, len(c):] = c[-1]
    return {
        "mean_curve": arr.mean(axis=0), "sd_curve": arr.std(axis=0),
        "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals)),
        "actions_mean": float(np.mean(lengths)), "actions_sd": float(np.std(lengths)),
    }

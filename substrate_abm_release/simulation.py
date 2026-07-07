"""
simulation.py
=============
Two-phase cooperative-diagnosis simulation (Okabe/Kanno CTW 2025), Phase-1 build.

Phase A (observation): each agent looks at its assigned observable nodes.
Phase B (cooperation): a randomly chosen agent resolves a metacognitive conflict
  via look / inform / query, until no conflicts remain, or 5 idle steps, or a
  100-action cap.

Dependent variable: team accuracy = mean over agents of (correct diagnoses / N).
Tracked per action so we can plot the Okabe/Kanno-style curve.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import random
import numpy as np

from bbn_engine import CausalSpec, BBNEngine, InferenceEngine, UNCERTAIN
from embm_agent import EMBMAgent, TYPE_I, TYPE_II, TYPE_III, _proj_key


# --------------------------------------------------------------------------- #
# Node-importance selection (Okabe/Kanno Eq. 3): sum of pairwise mutual
# information between a node and all others, under the agent's L1 beliefs.
# Computed once from the spec's priors (it is a structural property of the net).
# --------------------------------------------------------------------------- #
def _entropy(dist: np.ndarray) -> float:
    p = dist[dist > 0]
    return float(-(p * np.log2(p)).sum())


def node_importance(spec: CausalSpec) -> Dict[str, float]:
    """Approximate Eq. 3: importance(X) = sum_{Y != X} I(X;Y) under the prior.
    Computed with a clean BBN over the priors (no evidence)."""
    eng = BBNEngine(spec)
    names = list(spec.nodes)
    # marginal entropies
    H = {n: _entropy(np.array(list(eng.posterior(n).values()))) for n in names}
    imp = {n: 0.0 for n in names}
    for i, X in enumerate(names):
        for Y in names:
            if X == Y:
                continue
            # I(X;Y) = H(X) - H(X|Y); approximate H(X|Y) by averaging H(X|Y=y) over P(Y)
            pY = eng.posterior(Y)
            cond = 0.0
            for y_state, py in pY.items():
                if py <= 0:
                    continue
                e2 = BBNEngine(spec)
                e2.ingest(Y, y_state)
                cond += py * _entropy(np.array(list(e2.posterior(X).values())))
            imp[X] += max(0.0, H[X] - cond)
    return imp


# --------------------------------------------------------------------------- #
# Scenario = a ground-truth assignment of every node to a true state.
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    truth: Dict[str, str]               # node -> true state
    name: str = "scenario"


def sample_scenario(spec: CausalSpec, rng: random.Random, name="s") -> Scenario:
    """Forward-sample a consistent assignment from the network (ancestral sampling)."""
    truth: Dict[str, str] = {}
    eng = BBNEngine(spec)
    for n in spec.topo_order():
        # P(n | sampled parents) -- ingest the sampled parent states then read posterior
        e = BBNEngine(spec)
        for p in spec.parents.get(n, []):
            e.ingest(p, truth[p])
        post = e.posterior(n)
        states, probs = zip(*post.items())
        truth[n] = rng.choices(states, weights=probs, k=1)[0]
    return Scenario(truth, name)


# --------------------------------------------------------------------------- #
# Team accuracy
# --------------------------------------------------------------------------- #
def team_accuracy(agents: List[EMBMAgent], scenario: Scenario) -> float:
    N = len(agents[0].spec.nodes)
    accs = []
    for ag in agents:
        correct = 0
        for node, true_state in scenario.truth.items():
            if ag.M.diagnosis(node, ag.thd) == true_state:
                correct += 1
        accs.append(correct / N)
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    thd: float = 0.75
    gamma: float = 0.95
    max_actions: int = 100
    idle_limit: int = 5
    engine_factory: Callable[[CausalSpec], InferenceEngine] = BBNEngine  # SUBSTRATE HOOK


def build_team(spec: CausalSpec, observability: Dict[str, List[str]],
               cfg: SimConfig) -> List[EMBMAgent]:
    names = list(observability.keys())
    agents = []
    for nm in names:
        mates = [m for m in names if m != nm]
        agents.append(EMBMAgent(
            name=nm, teammates=mates, spec=spec,
            engine_factory=cfg.engine_factory,
            observable=observability[nm], thd=cfg.thd, gamma=cfg.gamma,
        ))
    return agents


def run_simulation(spec: CausalSpec, observability: Dict[str, List[str]],
                   scenario: Scenario, cfg: SimConfig, rng: random.Random
                   ) -> Tuple[List[float], List[EMBMAgent]]:
    """Run one full simulation; return (accuracy_curve, agents).
    accuracy_curve[k] = team accuracy after k actions (k=0 is pre-observation)."""
    agents = build_team(spec, observability, cfg)
    by_name = {a.name: a for a in agents}
    importance = node_importance(spec)
    curve = [team_accuracy(agents, scenario)]

    # ---- Phase A: observation (each agent looks at its assigned nodes) ----
    obs_actions = [(a, node) for a in agents for node in a.observable]
    rng.shuffle(obs_actions)
    for ag, node in obs_actions:
        ag.observe(node, scenario.truth[node])
        # teammates overhear nothing in the pure-observation phase (Okabe/Kanno)
        curve.append(team_accuracy(agents, scenario))

    # ---- Phase B: cooperation ----
    def diag_state(ags):
        """Hashable snapshot of all agents' L1 diagnoses -- the 'cognitive state'
        whose stability triggers termination (Okabe/Kanno: 5 idle steps)."""
        return tuple(
            tuple(a.M.diagnosis(n, cfg.thd) for n in spec.nodes) for a in ags
        )

    idle = 0
    last_state = diag_state(agents)
    while len(curve) - 1 < cfg.max_actions:
        # global termination: no agent has any conflict at all
        if all(not a.find_conflicts() for a in agents):
            break
        actor = rng.choice(agents)
        conflicts = actor.find_conflicts()
        if not conflicts:
            continue
        # pick the most important conflicting node
        node, ctype, mate = max(conflicts, key=lambda c: importance[c[0]])
        _resolve(actor, by_name, node, ctype, mate, scenario, cfg)
        curve.append(team_accuracy(agents, scenario))
        # idle = consecutive actions that did not change any L1 diagnosis
        now = diag_state(agents)
        if now == last_state:
            idle += 1
            if idle >= cfg.idle_limit:
                break
        else:
            idle = 0
            last_state = now

    return curve, agents


def _resolve(actor: EMBMAgent, by_name: Dict[str, EMBMAgent],
             node: str, ctype: str, mate: str, scenario: Scenario, cfg: SimConfig):
    """Resolve one conflict (Okabe/Kanno Table 3 logic, unified):
       - if self is UNCERTAIN about the node: QUERY the relevant teammate
       - elif self is certain but believes the teammate is UNCERTAIN/wrong: INFORM
       - elif the node is observable by self: LOOK-AT to settle it
       - else: INFORM (best available)."""
    self_diag = actor.M.diagnosis(node, actor.thd)
    teammate = by_name[mate]

    if self_diag == UNCERTAIN:
        # QUERY: ask the teammate for their diagnosis
        answer = teammate.answer_query(node)
        actor.hear_inform(mate, node, answer)
        # the third agent overhears the answer
        for other in actor.teammates:
            if other != mate:
                by_name[other].hear_inform(mate, node, answer)
        # querying also informs the teammate of self's UNCERTAIN status (they note the gap)
        teammate.M_proj[_proj_key(actor.name, actor.name)].ingest(
            node, teammate._soft_dist(node, answer) or {})
    elif node in actor.observable:
        # LOOK-AT: settle it directly with a fresh observation
        actor.observe(node, scenario.truth[node])
    else:
        # INFORM: push self's diagnosis to the teammate (and the third overhears)
        teammate.hear_inform(actor.name, node, self_diag)
        for other in actor.teammates:
            if other != mate:
                by_name[other].hear_inform(actor.name, node, self_diag)
        # self updates its own L3: it now believes the teammate believes self's diagnosis
        actor.M_proj[_proj_key(mate, actor.name)].ingest(
            node, actor._soft_dist(node, self_diag) or {})


# --------------------------------------------------------------------------- #
# Monte-Carlo runner: many runs, aligned curves, mean + SD (Okabe/Kanno style)
# --------------------------------------------------------------------------- #
def run_many(spec, observability, scenario, cfg, n_runs=100, seed=0):
    rng = random.Random(seed)
    curves = []
    finals = []
    lengths = []
    for r in range(n_runs):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        curve, _ = run_simulation(spec, observability, scenario, cfg, sub)
        curves.append(curve)
        finals.append(curve[-1])
        lengths.append(len(curve) - 1)  # number of actions
    # align by padding short curves with their final value (they have converged)
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
        arr[i, len(c):] = c[-1]
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)
    return {
        "mean_curve": mean,
        "sd_curve": sd,
        "final_mean": float(np.mean(finals)),
        "final_sd": float(np.std(finals)),
        "actions_mean": float(np.mean(lengths)),
        "actions_sd": float(np.std(lengths)),
    }

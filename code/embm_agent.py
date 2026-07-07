"""
embm_agent.py
=============
The EMBM (Extended Mutual Belief Model) agent: three cognitive layers realised
as 7 InferenceEngine instances, plus metacognitive conflict detection and the
look/inform/query action repertoire.

Layer structure for agent A on a 3-person team {A, B, C}:
  L1 (self):       M_A                      -- 1 engine
  L2 (direct):     M_A'(B), M_A'(C)         -- 2 engines (A's belief re each teammate's diagnosis)
  L3 (projected):  M_A''(B,A), M_A''(B,C),
                   M_A''(C,A), M_A''(C,B)   -- 4 engines (A's belief re a teammate's belief)

This mirrors Okabe/Kanno Fig. 1 exactly (no mental subgrouping, as in their model).

PHASE 1 SCOPE: every division is a BBNEngine and all agents share the substrate,
so all translation is identity. The hooks for substrate heterogeneity (Phases 2-4)
are marked with `# SUBSTRATE HOOK`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from itertools import permutations

from bbn_engine import CausalSpec, BBNEngine, InferenceEngine, UNCERTAIN

# Conflict types from Okabe/Kanno Table 3
TYPE_I = "I"     # M(node) vs M'(teammate, node)
TYPE_II = "II"   # M(node) vs M''(teammate, A, node)   -- belief of teammate about self
TYPE_III = "III" # M(node) vs M''(teammate, other, node) -- belief of teammate about the third


def _proj_key(believer: str, about: str) -> str:
    """Key for a projected (L3) division: 'A's belief about <believer>'s belief
    about <about>'s cognition'."""
    return f"{believer}|{about}"


@dataclass
class EMBMAgent:
    name: str
    teammates: List[str]                 # the other two names, e.g. ['B','C'] for A
    spec: CausalSpec
    engine_factory: Callable[[CausalSpec], InferenceEngine] = BBNEngine  # A's OWN substrate (L1)
    observable: List[str] = field(default_factory=list)  # nodes this agent can look at
    thd: float = 0.75
    gamma: float = 0.95                  # inter/intra-layer influence weight (Okabe/Kanno)

    # ---- Phase 4 substrate-heterogeneity configuration ----
    # teammate_factories: the TRUE engine factory each teammate actually runs.
    #   Used when this agent is substrate-AWARE (models teammates correctly).
    #   If None, defaults to this agent's own factory (homogeneous team = Phase 1).
    teammate_factories: Optional[Dict[str, Callable[[CausalSpec], InferenceEngine]]] = None
    # projection_mode: how A builds its models of teammates (L2 + L3 divisions)
    #   "aware" -> use the teammate's TRUE substrate (teammate_factories)
    #   "naive" -> use A's OWN substrate for every teammate model (Anthropomorphization Trap)
    projection_mode: str = "aware"

    def __post_init__(self):
        # resolve teammate factories (default: everyone runs my substrate = homogeneous)
        if self.teammate_factories is None:
            self.teammate_factories = {t: self.engine_factory for t in self.teammates}

        def model_factory(teammate: str) -> Callable[[CausalSpec], InferenceEngine]:
            """Which engine A uses to MODEL `teammate` (its L2/L3 divisions about them)."""
            if self.projection_mode == "naive":
                return self.engine_factory          # A assumes teammate reasons like A
            return self.teammate_factories[teammate]  # A models teammate's true substrate

        # L1: A's own cognition, in A's own substrate
        self.M = self.engine_factory(self.spec)

        # L2: A's belief about each teammate's diagnosis, in the model substrate
        self.M_direct: Dict[str, InferenceEngine] = {
            t: model_factory(t)(self.spec) for t in self.teammates
        }

        # L3: A's belief about teammate b's belief about target's cognition.
        # The believer is b, so the substrate is b's model substrate.
        others = [self.name] + self.teammates
        self.M_proj: Dict[str, InferenceEngine] = {}
        for b in self.teammates:
            for target in others:
                if target != b:
                    self.M_proj[_proj_key(b, target)] = model_factory(b)(self.spec)

    # ------------------------------------------------------------------ #
    # Evidence intake
    # ------------------------------------------------------------------ #
    def observe(self, node: str, true_state: str):
        """LOOK-AT: hard evidence into L1, with downward propagation to L2/L3
        (Okabe/Kanno downward effect: a new observation updates what A believes
        others should come to believe, entered as soft 'gamma' evidence)."""
        self.M.ingest(node, true_state)
        self._downward(node, true_state)

    def hear_inform(self, sender: str, node: str, message, sender_kind: str = None,
                    charitable: bool = False, sender_engine=None):
        """A receives `sender` informing about `node`. `message` is a
        translation.Message (sender's label + optional posterior). The message is
        translated tau_{sender_kind -> receiver_division_kind} into EACH receiving
        division according to THAT division's substrate -- this is where
        Translational Friction enters the simulation.

        Phase-1 compatibility: if `message` is a bare label/dict and sender_kind
        is None, behaves as the old identity-ingest path.
        """
        from translation import translate, KIND_PROB, KIND_TREE  # local import avoids cycle

        # --- Phase 1 / legacy path: bare diagnosis, no translation -----------
        if sender_kind is None:
            ev = self._as_evidence(message)
            if ev is None:
                return
            self.M_direct[sender].ingest(node, ev)
            for other in self.teammates:
                if other != sender:
                    self.M_direct[other].ingest(node, self._soft_dist(node, ev))
            self.M_proj[_proj_key(sender, self.name)].ingest(node, self._soft_dist(node, ev))
            self.M.ingest(node, self._soft_dist(node, ev))
            return

        # --- Phase 4 path: translate into each division per its substrate ----
        def kind_of(engine) -> str:
            return KIND_TREE if engine.__class__.__name__ == "DecisionTreeEngine" else KIND_PROB

        # 1) A's L2 model of the sender: translate sender -> (that model's kind)
        recv = self.M_direct[sender]
        ev = translate(message, sender_kind, kind_of(recv),
                       sender_engine=sender_engine, charitable=charitable)
        if ev is not None:
            recv.ingest(node, ev)

        # 2) parallel effect: A's model of the OTHER teammate also updates (overheard)
        for other in self.teammates:
            if other != sender:
                recv_o = self.M_direct[other]
                ev_o = translate(message, sender_kind, kind_of(recv_o),
                                 sender_engine=sender_engine, charitable=charitable)
                if ev_o is not None:
                    recv_o.ingest(node, self._maybe_soft(node, ev_o))

        # 3) A's L3 belief that the sender now believes A holds this
        proj = self.M_proj[_proj_key(sender, self.name)]
        ev_p = translate(message, sender_kind, kind_of(proj),
                         sender_engine=sender_engine, charitable=charitable)
        if ev_p is not None:
            proj.ingest(node, self._maybe_soft(node, ev_p))

        # 4) upward effect into A's OWN L1 (A's substrate)
        ev_self = translate(message, sender_kind, kind_of(self.M),
                            sender_engine=sender_engine, charitable=charitable)
        if ev_self is not None:
            self.M.ingest(node, self._maybe_soft(node, ev_self))

    def _maybe_soft(self, node, ev):
        """Apply gamma-softening to a label/dist for inter/intra-layer (non-primary)
        ingests, preserving the Phase-1 behaviour. Hard labels become soft dists;
        dicts are damped. A tree division will argmax-collapse a dict on ingest."""
        soft = self._soft_dist(node, ev)
        return soft if soft is not None else ev

    def answer_query(self, node: str):
        """A is queried about `node`; returns its current L1 diagnosis."""
        return self.M.diagnosis(node, self.thd)

    # ------------------------------------------------------------------ #
    # Metacognitive monitoring: find conflicts
    # ------------------------------------------------------------------ #
    def find_conflicts(self) -> List[Tuple[str, str, str]]:
        """Return list of (node, conflict_type, teammate) where A perceives a
        mismatch between layers. These drive action generation."""
        conflicts = []
        for node in self.spec.nodes:
            self_diag = self.M.diagnosis(node, self.thd)
            for t in self.teammates:
                # Type I: self vs A's direct belief about teammate t
                direct = self.M_direct[t].diagnosis(node, self.thd)
                if self._mismatch(self_diag, direct):
                    conflicts.append((node, TYPE_I, t))
                # Type II: self vs A's belief about t's belief about A
                projII = self.M_proj[_proj_key(t, self.name)].diagnosis(node, self.thd)
                if self._mismatch(self_diag, projII):
                    conflicts.append((node, TYPE_II, t))
                # Type III: self vs A's belief about t's belief about the OTHER teammate
                for other in self.teammates:
                    if other != t:
                        projIII = self.M_proj[_proj_key(t, other)].diagnosis(node, self.thd)
                        if self._mismatch(self_diag, projIII):
                            conflicts.append((node, TYPE_III, t))
        return conflicts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _downward(self, node: str, state: str):
        """Downward effect (Okabe/Kanno): L1 update flows to L2 and L3 as the
        belief that, once shared, others would hold this too. Entered softly."""
        soft = self._soft_dist(node, state)
        if soft is None:
            return
        for t in self.teammates:
            self.M_direct[t].ingest(node, soft)
        for eng in self.M_proj.values():
            eng.ingest(node, soft)

    def _soft_dist(self, node: str, evidence):
        """Convert a (possibly hard) diagnosis into a soft gamma-weighted
        distribution over `node`'s states, modelling that inter/intra-layer
        influence is strong (gamma) but not certain. Returns a dict or None."""
        states = self.spec.nodes[node]
        if isinstance(evidence, str):
            if evidence == UNCERTAIN:
                return None
            asserted = evidence
            rest = (1.0 - self.gamma) / max(1, len(states) - 1)
            return {s: (self.gamma if s == asserted else rest) for s in states}
        if isinstance(evidence, dict):
            # already a distribution; damp toward uniform by gamma
            unif = 1.0 / len(states)
            return {s: self.gamma * evidence.get(s, 0.0) + (1 - self.gamma) * unif
                    for s in states}
        return None

    def _as_evidence(self, diag):
        if diag == UNCERTAIN or diag is None:
            return None
        return diag

    @staticmethod
    def _mismatch(a, b) -> bool:
        """A conflict exists when the two diagnoses are both committed and differ,
        OR one is committed and the other UNCERTAIN (Okabe/Kanno treat the
        information gap as a conflict to be resolved)."""
        if a == b:
            return False
        return True

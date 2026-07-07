"""
tree_engine.py
==============
Phase 2, Engine C: a decision-tree reasoner implementing the same
InferenceEngine interface from the same CausalSpec.

DESIGN (architecture spec sec. 2.1, Engine C):
A decision tree maps an observed-feature vector to a diagnosis of a TARGET node.
Unlike the BBN/Markov engines it has NO causal-inference-in-arbitrary-directions:
it can only diagnose node N from whatever OTHER node-values it currently holds as
features. We therefore build, for each node N, a small classifier:

    features(N) = all other nodes
    tree(N): partial feature assignment -> distribution over N's states

To keep domain knowledge IDENTICAL across engines (the shared-spec control), the
tree is COMPILED FROM THE SPEC, not learned from sampled data: for a given set of
held feature values we compute the exact BBN posterior of N conditioned on those
features, and that is the tree's leaf distribution. Conceptually this is a tree
whose splits are the observed features and whose leaves carry class probabilities
(the leaf pseudo-posterior, architecture sec. 2.1: "leaf class-probabilities as a
pseudo-posterior" so THD and soft evidence remain meaningful).

THE PRINCIPLED PROPERTIES that distinguish the tree substrate:
  (1) It reasons ONLY from currently-held feature values. With no features set,
      it returns the node's marginal prior (a leaf at the root).
  (2) It does NOT maintain a coherent joint: updating belief about N does not
      propagate to other nodes (no belief propagation). Each node is diagnosed
      independently from the features it happens to hold. This is the tree's
      structural signature -- locality without joint coherence.
  (3) Its committed output is a crisp label (argmax) with a leaf probability;
      this is what makes the ->tree translation lossy (uncertainty discarded on
      commit) and tree-> translation a false-confidence injection.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from bbn_engine import CausalSpec, BBNEngine, InferenceEngine, UNCERTAIN


class DecisionTreeEngine(InferenceEngine):
    def __init__(self, spec: CausalSpec):
        self.spec = spec
        self.names = list(spec.nodes)
        # held feature values: node -> state label (a "feature" the tree currently has)
        self._features: Dict[str, str] = {}
        # cache of compiled leaf posteriors keyed by (target, frozenset(features))
        self._cache: Dict[tuple, np.ndarray] = {}

    # ---- interface --------------------------------------------------------
    def ingest(self, node: str, evidence) -> None:
        """A decision tree only holds crisp feature values. Soft evidence is
        collapsed to its argmax on intake -- the tree cannot represent a
        distribution as a held feature. (UNCERTAIN holds no feature.)"""
        if isinstance(evidence, str):
            if evidence == UNCERTAIN:
                return
            self._features[node] = evidence
        elif isinstance(evidence, dict):
            # collapse to argmax: the tree's representational vocabulary is crisp
            if not evidence:
                return
            best = max(evidence.items(), key=lambda kv: kv[1])
            if best[1] <= 0:
                return
            self._features[node] = best[0]
        else:
            raise TypeError("evidence must be a state label or dict")

    def posterior(self, node: str) -> Dict[str, float]:
        dist = self._leaf(node)
        return {s: float(p) for s, p in zip(self.spec.nodes[node], dist)}

    def diagnosis(self, node: str, thd: float):
        post = self.posterior(node)
        state, p = max(post.items(), key=lambda kv: kv[1])
        return state if p >= thd else UNCERTAIN

    def snapshot(self) -> dict:
        return {"features": dict(self._features)}

    def restore(self, snap: dict) -> None:
        self._features = dict(snap["features"])

    # ---- compiled-from-spec leaf distribution -----------------------------
    def _leaf(self, target: str) -> np.ndarray:
        # features available for this target = all held features except the target itself
        feats = {k: v for k, v in self._features.items() if k != target}
        key = (target, frozenset(feats.items()))
        if key in self._cache:
            return self._cache[key]
        # compile the leaf: exact posterior of target given the held feature values
        e = BBNEngine(self.spec)
        for fnode, fstate in feats.items():
            e.ingest(fnode, fstate)
        dist = np.array(list(e.posterior(target).values()))
        self._cache[key] = dist
        return dist

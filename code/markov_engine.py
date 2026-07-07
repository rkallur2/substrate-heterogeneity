"""
markov_engine.py
================
Phase 2, Engine B: a first-order Markov reasoner over the causal SKELETON,
implementing the same InferenceEngine interface as BBNEngine from the same
CausalSpec.

DESIGN (architecture spec sec. 2.1, Engine B; redesigned after validation):
The original domain is a causal DAG in which a node may have several parents.
A first-order Markov model keeps, for each node, only its DOMINANT parent --
the single parent that most reduces uncertainty about the node -- and builds a
tree (a polytree with in-degree <= 1) along those links. Inference is exact
sum-product belief propagation on that tree.

THE PRINCIPLED LOSS: for any node with >1 true parent, the secondary parents'
influence is discarded. Evidence still propagates along the dominant causal
skeleton (unlike the naive topological chain, which severed most links), but
multi-parent interactions -- explaining-away, joint dependence -- are lost.
This is the nameable, defensible approximation that distinguishes the Markov
substrate from the BBN, and it is precisely the kind of structure the
Translational-Friction story concerns. Choice of "dominant parent" (by mutual
information under the priors) is a documented modeling commitment.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from bbn_engine import CausalSpec, BBNEngine, InferenceEngine, UNCERTAIN


def _norm(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class MarkovEngine(InferenceEngine):
    def __init__(self, spec: CausalSpec):
        self.spec = spec
        self.names = list(spec.nodes)
        self.card = {n: len(spec.nodes[n]) for n in self.names}

        ref = BBNEngine(spec)
        self.prior = {n: np.array(list(ref.posterior(n).values())) for n in self.names}

        # --- choose each node's dominant parent (max mutual information) ---
        self.dom_parent: Dict[str, Optional[str]] = {}
        for n in self.names:
            pa = spec.parents.get(n, [])
            if not pa:
                self.dom_parent[n] = None
                continue
            best, best_mi = None, -1.0
            Hn = _entropy(self.prior[n])
            for p in pa:
                # I(n;p) = H(n) - H(n|p), H(n|p) averaged over p's prior
                cond = 0.0
                for s_i, s in enumerate(spec.nodes[p]):
                    e = BBNEngine(spec); e.ingest(p, s)
                    cond += self.prior[p][s_i] * _entropy(np.array(list(e.posterior(n).values())))
                mi = max(0.0, Hn - cond)
                if mi > best_mi:
                    best_mi, best = mi, p
            self.dom_parent[n] = best

        # --- transition P(node | dominant parent) from the spec ---
        # T[node] : shape (card[parent], card[node]); for roots, None
        self.T: Dict[str, Optional[np.ndarray]] = {}
        for n in self.names:
            p = self.dom_parent[n]
            if p is None:
                self.T[n] = None
                continue
            M = np.zeros((self.card[p], self.card[n]))
            for s_i, s in enumerate(spec.nodes[p]):
                e = BBNEngine(spec); e.ingest(p, s)
                M[s_i, :] = np.array(list(e.posterior(n).values()))
            self.T[n] = M

        # --- build children adjacency of the skeleton tree ---
        self.children: Dict[str, List[str]] = {n: [] for n in self.names}
        self.roots: List[str] = []
        for n in self.names:
            p = self.dom_parent[n]
            if p is None:
                self.roots.append(n)
            else:
                self.children[p].append(n)

        # evidence potentials
        self._pot: Dict[str, np.ndarray] = {}

    # ---- interface --------------------------------------------------------
    def ingest(self, node: str, evidence) -> None:
        states = self.spec.nodes[node]
        if isinstance(evidence, str):
            if evidence == UNCERTAIN:
                return
            vec = np.zeros(len(states))
            vec[self.spec.state_index(node, evidence)] = 1.0
            self._pot[node] = vec
        elif isinstance(evidence, dict):
            vec = np.array([evidence.get(s, 0.0) for s in states], dtype=float)
            if vec.sum() <= 0:
                return
            vec = _norm(vec)
            if node in self._pot:
                comb = self._pot[node] * vec
                if comb.sum() > 0:
                    vec = _norm(comb)
            self._pot[node] = vec
        else:
            raise TypeError("evidence must be a state label or dict")

    def posterior(self, node: str) -> Dict[str, float]:
        marg = self._propagate()
        return {s: float(p) for s, p in zip(self.spec.nodes[node], marg[node])}

    def diagnosis(self, node: str, thd: float):
        post = self.posterior(node)
        state, p = max(post.items(), key=lambda kv: kv[1])
        return state if p >= thd else UNCERTAIN

    def snapshot(self) -> dict:
        return {"pot": {k: v.copy() for k, v in self._pot.items()}}

    def restore(self, snap: dict) -> None:
        self._pot = {k: v.copy() for k, v in snap["pot"].items()}

    # ---- sum-product on the skeleton tree (polyforest) --------------------
    def _pot_of(self, n: str) -> np.ndarray:
        return self._pot.get(n, np.ones(self.card[n]))

    def _propagate(self) -> Dict[str, np.ndarray]:
        # Belief propagation on a forest of trees (each node has <=1 parent).
        # Two passes: upward (leaves->roots) then downward (roots->leaves).

        # upward messages: m_up[child] is a factor over the PARENT's states,
        # summarizing the subtree rooted at child.
        m_up: Dict[str, np.ndarray] = {}

        order = self._post_order()  # children before parents
        for n in order:
            # collect messages already computed from n's own children
            belief_n = self._pot_of(n).astype(float).copy()
            for c in self.children[n]:
                belief_n = belief_n * m_up[c]  # message from child c over n's states
            p = self.dom_parent[n]
            if p is not None:
                # send upward to parent: sum_n  P(n|p) * belief_n  -> vector over p's states
                T = self.T[n]                       # (card[p], card[n])
                m_up[n] = _norm(T @ belief_n)        # over parent states
            # store partial belief for downward pass
            m_up.setdefault("_belief_" + n, None)
            m_up["_belief_" + n] = belief_n

        # downward pass: combine with message from parent
        marg: Dict[str, np.ndarray] = {}
        m_down: Dict[str, np.ndarray] = {}
        pre = list(reversed(order))  # parents before children
        for n in pre:
            belief_n = m_up["_belief_" + n].copy()
            p = self.dom_parent[n]
            if p is None:
                belief_n = belief_n * self.prior[n]
                marg[n] = _norm(belief_n)
            else:
                # message from parent down to n: sum_p P(n|p) * [parent belief excluding n's upward msg]
                parent_belief = marg[p].copy()
                # divide out n's own upward contribution to avoid double counting
                contrib = m_up[n]
                safe = np.where(contrib > 1e-12, contrib, 1.0)
                parent_minus = _norm(parent_belief / safe)
                T = self.T[n]                       # (card[p], card[n])
                down = parent_minus @ T             # over n's states
                marg[n] = _norm(belief_n * down)
        return marg

    def _post_order(self) -> List[str]:
        order, seen = [], set()
        def visit(n):
            if n in seen: return
            seen.add(n)
            for c in self.children[n]:
                visit(c)
            order.append(n)  # n after its children
        for r in self.roots:
            visit(r)
        # include any nodes not reached (shouldn't happen in a forest)
        for n in self.names:
            if n not in seen:
                visit(n)
        return order

"""
bbn_engine.py
=============
Phase 1 inference substrate: a discrete Bayesian Belief Network with exact
inference by variable elimination, implemented from scratch (no pgmpy).

This module defines:
  - CausalSpec : the shared, declarative domain description (see architecture spec sec. 3)
  - InferenceEngine : the abstract interface every cognitive division uses (sec. 2)
  - BBNEngine : the Bayesian-network implementation of that interface (Engine A)

The InferenceEngine interface is deliberately tiny (ingest / posterior / diagnosis).
Phases 2-4 add MarkovEngine and DecisionTreeEngine implementing the SAME interface,
so the EMBM control logic never has to change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np

UNCERTAIN = "UNCERTAIN"


# ---------------------------------------------------------------------------
# Shared causal spec  (architecture sec. 3 -- the control object)
# ---------------------------------------------------------------------------
@dataclass
class CausalSpec:
    """A single declarative description of the domain that EVERY engine is
    built from. Holding this constant across BBN / Markov / tree is what makes
    the substrate comparison clean.

    nodes:   ordered dict-like  name -> list of state labels
    parents: name -> list of parent names (defines the DAG edges)
    cpts:    name -> np.ndarray giving P(node | parents).
             Shape = (|parent0|, |parent1|, ..., |node|); last axis is the node.
             For a root node, shape = (|node|,) and it is the prior.
    """
    nodes: Dict[str, List[str]]
    parents: Dict[str, List[str]]
    cpts: Dict[str, np.ndarray]

    def __post_init__(self):
        # ---- validation: catch spec errors early, they are murder to debug later
        for n, states in self.nodes.items():
            assert len(states) == len(set(states)), f"duplicate states in {n}"
            pa = self.parents.get(n, [])
            cpt = self.cpts[n]
            expected_shape = tuple(len(self.nodes[p]) for p in pa) + (len(states),)
            assert cpt.shape == expected_shape, (
                f"CPT shape mismatch for {n}: got {cpt.shape}, expected {expected_shape}"
            )
            # each conditional distribution must sum to 1 over the node's own axis
            sums = cpt.sum(axis=-1)
            assert np.allclose(sums, 1.0), f"CPT for {n} not normalised (sums={sums})"

    def topo_order(self) -> List[str]:
        """Topological order of nodes (parents before children)."""
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            for p in self.parents.get(n, []):
                visit(p)
            seen.add(n)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order

    def state_index(self, node: str, state: str) -> int:
        return self.nodes[node].index(state)


# ---------------------------------------------------------------------------
# The interface  (architecture sec. 2)
# ---------------------------------------------------------------------------
class InferenceEngine:
    """Every cognitive division (the 7 EMBM belief structures per agent) is one
    of these. Phases 2-4 supply MarkovEngine / DecisionTreeEngine with the same
    three methods."""

    def ingest(self, node: str, evidence) -> None:
        """Incorporate evidence about one node. `evidence` is either:
           - a state label (HARD evidence: that node is certainly in that state), or
           - a dict {state: prob} (SOFT evidence: a distribution)."""
        raise NotImplementedError

    def posterior(self, node: str) -> Dict[str, float]:
        """Current belief distribution over the node's states."""
        raise NotImplementedError

    def diagnosis(self, node: str, thd: float):
        """Threshold the posterior into a committed state, or UNCERTAIN."""
        raise NotImplementedError

    def snapshot(self) -> dict:
        """Cheap copy of internal evidence state, for cloning beliefs."""
        raise NotImplementedError

    def restore(self, snap: dict) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Engine A : Bayesian Belief Network with variable elimination
# ---------------------------------------------------------------------------
class _Factor:
    """A discrete factor over a set of variables, stored as an ndarray whose
    axes correspond, in order, to `vars`. Cardinalities come from the spec."""

    __slots__ = ("vars", "values")

    def __init__(self, variables: List[str], values: np.ndarray):
        self.vars = list(variables)
        self.values = values

    def __repr__(self):
        return f"Factor({self.vars}, shape={self.values.shape})"


class BBNEngine(InferenceEngine):
    """Exact discrete Bayesian network inference.

    Evidence is stored as a list of (node, kind, payload) entries; posteriors
    are computed on demand by variable elimination. Soft evidence is handled by
    attaching a likelihood (virtual-evidence) factor on the node.
    """

    def __init__(self, spec: CausalSpec):
        self.spec = spec
        # hard evidence: node -> state label
        self._hard: Dict[str, str] = {}
        # soft evidence (likelihoods): node -> np.ndarray over node's states
        self._soft: Dict[str, np.ndarray] = {}
        self._cache: Dict = {}          # (evidence_key, query) -> dist
        self._ev_key = ()               # current evidence fingerprint

    def _refresh_key(self):
        self._ev_key = (
            tuple(sorted(self._hard.items())),
            tuple((k, tuple(np.round(v, 6))) for k, v in sorted(self._soft.items())),
        )

    # ---- interface --------------------------------------------------------
    def ingest(self, node: str, evidence) -> None:
        if isinstance(evidence, str):
            if evidence == UNCERTAIN:
                return  # an explicit "I don't know" carries no information here
            self._hard[node] = evidence
            self._soft.pop(node, None)  # hard supersedes any prior soft
        elif isinstance(evidence, dict):
            states = self.spec.nodes[node]
            vec = np.array([evidence.get(s, 0.0) for s in states], dtype=float)
            if vec.sum() <= 0:
                return
            vec = vec / vec.sum()
            # combine multiplicatively with any existing soft evidence
            if node in self._soft:
                vec = self._soft[node] * vec
                if vec.sum() <= 0:
                    return
                vec = vec / vec.sum()
            self._soft[node] = vec
        else:
            raise TypeError(f"evidence must be a state label or dict, got {type(evidence)}")
        self._refresh_key()

    def posterior(self, node: str) -> Dict[str, float]:
        key = (self._ev_key, node)
        cached = self._cache.get(key)
        if cached is None:
            dist = self._infer(node)
            self._cache[key] = dist
            cached = dist
        return {s: float(p) for s, p in zip(self.spec.nodes[node], cached)}

    def diagnosis(self, node: str, thd: float):
        post = self.posterior(node)
        state, p = max(post.items(), key=lambda kv: kv[1])
        return state if p >= thd else UNCERTAIN

    def snapshot(self) -> dict:
        return {
            "hard": dict(self._hard),
            "soft": {k: v.copy() for k, v in self._soft.items()},
        }

    def restore(self, snap: dict) -> None:
        self._hard = dict(snap["hard"])
        self._soft = {k: v.copy() for k, v in snap["soft"].items()}
        self._refresh_key()

    # ---- inference engine -------------------------------------------------
    def _base_factors(self) -> List[_Factor]:
        """One factor per node from its CPT, with hard evidence reducing factors
        and soft evidence added as likelihood factors."""
        factors: List[_Factor] = []
        for n in self.spec.nodes:
            pa = self.spec.parents.get(n, [])
            vars_ = pa + [n]
            f = _Factor(vars_, self.spec.cpts[n].astype(float).copy())
            factors.append(f)
        # soft-evidence likelihood factors
        for n, vec in self._soft.items():
            factors.append(_Factor([n], vec.copy()))
        # apply hard evidence by slicing every factor that mentions the node
        for n, state in self._hard.items():
            idx = self.spec.state_index(n, state)
            new_factors = []
            for f in factors:
                if n in f.vars:
                    ax = f.vars.index(n)
                    sliced = np.take(f.values, idx, axis=ax)
                    remaining = [v for v in f.vars if v != n]
                    new_factors.append(_Factor(remaining, sliced))
                else:
                    new_factors.append(f)
            factors = new_factors
        return factors

    @staticmethod
    def _multiply(f1: _Factor, f2: _Factor) -> _Factor:
        """Pointwise product of two factors over the union of their variables."""
        union = list(dict.fromkeys(f1.vars + f2.vars))

        def aligned(f: _Factor) -> np.ndarray:
            # current cardinalities of f's own vars
            card = {v: f.values.shape[i] for i, v in enumerate(f.vars)}
            # target shape: cardinality where present, else 1 (broadcast)
            target = [card.get(v, 1) for v in union]
            # build an array indexed in union order
            arr = f.values
            # permute f's existing axes into the order they appear within union
            present_order = [v for v in union if v in f.vars]
            perm = [f.vars.index(v) for v in present_order]
            arr = np.transpose(arr, perm) if perm else arr
            return arr.reshape(target)

        a = aligned(f1)
        b = aligned(f2)
        # resolve true cardinalities for the union (max over the two, since
        # broadcast dims are 1)
        out = a * b
        return _Factor(union, out)

    @staticmethod
    def _marginalize(f: _Factor, var: str) -> _Factor:
        ax = f.vars.index(var)
        summed = f.values.sum(axis=ax)
        remaining = [v for v in f.vars if v != var]
        return _Factor(remaining, summed)

    def _infer(self, query: str) -> np.ndarray:
        """Posterior distribution P(query | evidence) by variable elimination."""
        factors = self._base_factors()
        # eliminate every variable except the query
        elim = [v for v in self.spec.topo_order() if v != query]
        for var in elim:
            involved = [f for f in factors if var in f.vars]
            if not involved:
                continue
            prod = involved[0]
            for f in involved[1:]:
                prod = self._multiply(prod, f)
            summed = self._marginalize(prod, var)
            factors = [f for f in factors if var not in f.vars] + [summed]
        # multiply whatever remains; result should be over {query}
        result = factors[0]
        for f in factors[1:]:
            result = self._multiply(result, f)
        # collapse any stray singleton axes, reorder to query axis
        if query not in result.vars:
            # query was hard-evidenced; return a one-hot
            vec = np.zeros(len(self.spec.nodes[query]))
            vec[self.spec.state_index(query, self._hard[query])] = 1.0
            return vec
        ax = result.vars.index(query)
        vec = np.moveaxis(result.values, ax, 0)
        vec = vec.reshape(len(self.spec.nodes[query]), -1).sum(axis=1)
        total = vec.sum()
        return vec / total if total > 0 else np.ones_like(vec) / len(vec)

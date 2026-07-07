"""
translation.py
==============
Phase 3 (THE CORE CONTRIBUTION): the translation layer that operationalizes
"Translational Friction." When agent i (engine X) communicates a diagnosis to
agent j (engine Y), tau_{X->Y} maps the message into evidence j can ingest.

Every rule is DERIVED FROM ENGINE CAPABILITY, not a tuned noise parameter
(architecture sec. 4). The rules:

  prob -> prob  (BBN<->Markov): the message carries the sender's graded posterior;
                it transfers as soft evidence. Loss is downstream-structural
                (the receiver's engine may not exploit it richly), not in tau.

  prob -> tree  (BBN/Markov -> Tree): the tree has no soft-evidence slot, so the
                sender's distribution is COLLAPSED to a crisp label on receipt.
                Uncertainty is destroyed in the message. (Strict tau.)

  tree -> prob  (Tree -> BBN/Markov): the tree only ever held a crisp label (it
                discarded uncertainty at its own diagnosis step), so the message
                is a hard label and enters the probabilistic receiver as HARD
                evidence -- FALSE-CONFIDENCE INJECTION.

  X -> X        (matched): identity (the Okabe/Kanno baseline).

CHARITABLE-tau SWITCH (architecture sec. 4 robustness move): set charitable=True
to soften the two lossy directions --
  - prob->tree keeps the sender's full distribution as a soft feature instead of
    collapsing to argmax (tests whether the effect survives a generous rule);
  - tree->prob enters the tree's leaf pseudo-posterior as SOFT evidence instead of
    hard, so false confidence is not manufactured.
If the substrate-mismatch effect persists under charitable=True, it cannot be
dismissed as an artifact of an unfairly lossy translation.

A "message" here is what the sender chooses to communicate about one node:
either a committed state label (the sender's diagnosis) or, for probabilistic
senders, optionally the full posterior. We represent it as:
    Message(node, label, dist)
where label is the sender's argmax/diagnosis (possibly UNCERTAIN) and dist is the
sender's full posterior dict if the sender is probabilistic, else None.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from bbn_engine import UNCERTAIN


# engine "kind" tags
KIND_PROB = "prob"   # graded posterior, native soft evidence (BBN, Markov)
KIND_TREE = "tree"   # crisp label + leaf pseudo-posterior, no soft slot


@dataclass
class Message:
    node: str
    label: str                      # sender's committed diagnosis (or UNCERTAIN)
    dist: Optional[Dict[str, float]]  # sender's full posterior, if probabilistic; else None


def make_message(sender_engine, sender_kind: str, node: str, thd: float) -> Message:
    """Build the message a sender emits about `node`, from its own beliefs."""
    label = sender_engine.diagnosis(node, thd)
    dist = None
    if sender_kind == KIND_PROB:
        dist = sender_engine.posterior(node)  # full graded posterior available
    # a tree sender exposes only its committed label (its diagnosis); even though
    # it has a leaf pseudo-posterior, the *communicated* content is the crisp call
    # unless charitable translation chooses to expose it (handled in translate()).
    return Message(node, label, dist)


def translate(msg: Message, sender_kind: str, receiver_kind: str,
              sender_engine=None, charitable: bool = False):
    """Return the evidence object to ingest into the RECEIVER for msg.node,
    or None if nothing should be ingested (e.g. sender was UNCERTAIN).

    Evidence object is either a state label (hard) or a dict (soft), matching
    InferenceEngine.ingest's contract.
    """
    if msg.label == UNCERTAIN and msg.dist is None:
        # an uncertain crisp sender communicates only "I don't know" -> no evidence
        return None

    # ---- matched substrate: identity ------------------------------------
    if sender_kind == receiver_kind:
        if receiver_kind == KIND_PROB:
            # pass the graded posterior (or the label if that's all there is)
            return msg.dist if msg.dist is not None else msg.label
        else:  # tree -> tree: crisp label
            return None if msg.label == UNCERTAIN else msg.label

    # ---- prob -> prob (BBN <-> Markov): graded posterior as soft evidence -
    if sender_kind == KIND_PROB and receiver_kind == KIND_PROB:
        return msg.dist if msg.dist is not None else msg.label

    # ---- prob -> tree: distribution collapses on receipt -----------------
    if sender_kind == KIND_PROB and receiver_kind == KIND_TREE:
        if charitable and msg.dist is not None:
            # generous: let the tree take the full distribution (it will argmax
            # internally on ingest, but at least the dist is offered)
            return msg.dist
        # strict: uncertainty destroyed -> crisp label only
        return None if msg.label == UNCERTAIN else msg.label

    # ---- tree -> prob: false-confidence injection ------------------------
    if sender_kind == KIND_TREE and receiver_kind == KIND_PROB:
        if charitable and sender_engine is not None:
            # generous: enter the tree's leaf pseudo-posterior as SOFT evidence,
            # so no unwarranted certainty is manufactured
            soft = sender_engine.posterior(msg.node)
            return soft
        # strict: the crisp label enters as HARD evidence (false confidence)
        return None if msg.label == UNCERTAIN else msg.label

    raise ValueError(f"unhandled translation {sender_kind}->{receiver_kind}")

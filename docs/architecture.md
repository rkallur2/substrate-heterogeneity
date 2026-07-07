# Model Architecture: Inference-Substrate Heterogeneity and the Corruption of Recursive Mutual Beliefs

**Working spec for a team-cognition ABM extending Okabe, Kanno et al. (CTW 2025).**
Status: pre-implementation design. No results yet. Everything below is a build target, not a finding.

---

## 0. One-paragraph statement of what the model must show

Three agents cooperatively diagnose a system under partial observability, communicating to build team situation awareness (TSA), exactly as in Okabe/Kanno. The single new manipulation: agents may run **different inference engines** (the machinery that turns evidence into a belief distribution over part-states). We hold the EMBM recursive belief structure (self / direct / projected) and the task constant, and vary (a) whether teammates share an inference substrate and (b) whether each agent's projected-belief layer correctly models the partner's substrate or naively assumes homogeneity. The model must produce the Okabe/Kanno dependent variables — team accuracy over action sequence, actions-to-convergence, and variance — so results are directly comparable to their baseline curves.

---

## 1. What is held constant from Okabe/Kanno (the benchmark spine)

To keep results commensurable with the source paper, replicate these exactly:

- **Task.** A diagnosis problem over 17 nodes with discrete states (their car-fault domain, or a structurally equivalent synthetic causal system — see §7 on portability). 10 nodes observable by at least one agent; observability partitioned across agents per their Fig. 8 scheme (each agent sees 5 nodes, with overlaps).
- **EMBM cognitive layers.** Each agent holds 7 belief structures: 1 self-cognition (M), 2 direct beliefs (M′ about each teammate), 4 projected beliefs (M″). No mental subgrouping (they dropped it; so do we, for comparability).
- **Two phases.** Observation phase (each agent observes its assigned nodes; ~15 actions), then cooperation phase (look/inform/query driven by metacognitive conflict resolution; terminates on no-conflict, 5 idle steps, or 100-action cap).
- **Action types.** `look-at`, `inform`, `query`. One node's evidence per action. Third agent overhears communication.
- **Conflict-resolution logic** (their Table 3, Types I/II/III) and **node-importance selection** (their Eq. 3, mutual-information sum) — reused unchanged.
- **Threshold THD** for committing a diagnosis (sweep 0.75/0.80/0.85 as they did, for sensitivity).
- **Evaluation.** Team accuracy = mean over agents of (correctly diagnosed nodes / total). Track per-action; report mean and SD over 100 runs per scenario; use their 6 difficulty-ranked scenarios.

**Why hold all this constant:** the entire credibility of the contribution rests on the claim that any difference in the curves is attributable to substrate heterogeneity and *nothing else*. Every degree of freedom you reuse from Okabe/Kanno is a confound you've pre-closed. State this explicitly in the methods section.

---

## 2. The inference substrate as a pluggable component

This is the core abstraction. Define an **InferenceEngine interface** that every cognitive division (all 7 BBNs in the original) is replaced by. The interface has exactly three operations, because that's all the EMBM machinery ever asks of a belief structure:

```
InferenceEngine:
    ingest(node, evidence)        # incorporate hard or soft evidence about one node
    posterior(node) -> dist       # return current belief distribution over that node's states
    diagnosis(node, THD) -> state | UNCERTAIN   # threshold the posterior into a commitment
```

Everything the agent does — committing a diagnosis, detecting a conflict between layers, deciding to inform/query — is expressed through these three calls. If the interface is clean, you can swap engines without touching the EMBM control logic. **This separation is the methodological contribution; protect it.**

### 2.1 The three engines

Each implements the interface over the same 17-node domain, specified by hand from a **shared causal spec** (see §3) so domain knowledge is identical across engines.

**Engine A — Bayesian Belief Network (BBN).** The Okabe/Kanno original. Posterior = exact belief propagation over the causal DAG. Native graded posteriors; native soft evidence. This is your replication anchor — a homogeneous all-BBN team must reproduce their published curves, or your reimplementation is wrong. **Do this validation before anything else (§8, Phase 1).**

**Engine B — Hidden Markov / dynamic-Bayes-style probabilistic engine.** Hidden states = true node conditions; observations = agent percepts; "transitions" encode causal propagation between parts. Gives graded posteriors (forward inference), so soft evidence and THD work natively. The catch: a DAG of causes must be linearized/ordered into a chain-like structure, which is an approximation. **That approximation is a feature, not a bug** — it's a manipulable distortion and a finding in its own right ("how much does forcing causal structure into a Markov assumption change coordination dynamics?"). Document the linearization choice explicitly.

**Engine C — Decision tree.** Maps observed-feature vectors to a node diagnosis. Use **leaf class-probabilities as a pseudo-posterior** so THD and soft evidence remain meaningful — do NOT go fully crisp, or you change the cognitive model (no graded uncertainty) and lose comparability. Key conceptual property: a tree has **no causal-inference-in-any-direction** capacity. It can only infer an unobservable node if that node is a *target label* with observable features as inputs. So you pre-commit, per tree, which nodes are inferable targets. This limitation is real and you must surface it in methods — it's exactly the kind of thing a CTW/JASSS reviewer probes.

### 2.2 Specify, don't learn

Build all three engines by hand from the shared causal spec. Do **not** learn them from generated fault data. Reason: learned-model quality (sampling variance, hyperparameters, training-set size) would contaminate the substrate comparison with noise that has nothing to do with the substrate's *structure*. You want differences attributable to "BBN vs Markov vs tree as ways of representing the same knowledge," not "this tree happened to train badly." Hand-specification keeps domain knowledge a controlled constant.

---

## 3. The shared causal spec (the control object)

A single declarative description of the domain that all three engines are built from:

- Node set (17), each with its discrete state space.
- Causal edges (the DAG from the original).
- Prior probabilities per root node.
- Conditional probability tables per edge.

Each engine consumes this spec differently:
- BBN uses it directly (DAG + CPTs).
- Markov engine linearizes the DAG into an ordered chain and derives transition/emission matrices from the CPTs.
- Decision tree is compiled from the spec by enumerating evidence→diagnosis mappings (or by exhaustively expanding the CPTs into a decision structure for designated target nodes).

**This object is what makes the comparison clean.** If a reviewer asks "how do you know the engines encode the same knowledge?", the answer is "they're all derived from this one spec." Make the spec a first-class, version-controlled artifact.

---

## 4. The translation layer = Translational Friction, made computational

When agent *i* (engine X) communicates evidence to agent *j* (engine Y), the message must cross a representational gap. Define an explicit **translation rule** `τ_{X→Y}(evidence)`:

- **Matched substrates (X = Y):** identity. Evidence passes losslessly. This is the Okabe/Kanno case and your control condition.
- **Mismatched substrates (X ≠ Y):** lossy re-encoding. Examples of the loss you must define:
  - BBN graded soft evidence → decision tree: tree has no soft-evidence slot, so a distribution must collapse to a branch (crisp label or nearest leaf). **Information lost = the uncertainty.**
  - Markov forward-belief → BBN node evidence: re-expressed as soft evidence, but the chain-linearization may have entangled nodes the DAG kept separate. **Information distorted = cross-node dependence.**
  - Tree crisp label → BBN/Markov: enters as hard evidence even when the sender was actually uncertain (the tree threw the uncertainty away upstream). **Risk = false confidence injected into the receiver.**

**The translation layer is the contribution, not a nuisance parameter.** Resist the urge to make it "neutral." The whole point is that there is no neutral translation across incompatible substrates, and the coordination cost of that is what you're measuring. Define each `τ` explicitly, justify it, and treat the *choice* of translation rule as a documented modeling commitment (and ideally a robustness axis — try a charitable vs. a lossy τ and show the gap persists).

Operationalize **Translational Friction** as a measured quantity: the delta in team accuracy and in actions-to-convergence between a matched-substrate team and an otherwise-identical mismatched-substrate team. Friction becomes a number, per scenario, per THD.

---

## 5. The projected-belief-corruption mechanism = your distinction from CG-BAG and the ToM-order work

This is the part no neighbor has. Two sub-conditions, crossed with substrate matching:

**Substrate-aware projection.** Agent *i*'s projected-belief layer M″ correctly models that teammate *j* runs a different engine. When *i* predicts what *j* believes *i* believes, it applies the correct `τ` in its head. Its mutual beliefs stay calibrated despite the gap.

**Substrate-naive projection (the Anthropomorphization Trap).** Agent *i*'s M″ assumes *j* reasons the way *i* does (substrate homogeneity). When *i* forms beliefs about *j*'s beliefs, it uses its *own* engine as the model of *j*'s mind. Across a real substrate gap, *i*'s projected beliefs are now systematically wrong — confidently, in a specific, modelable direction.

**Prediction (the headline):** substrate-naive projection across a substrate gap should reproduce and *worsen* the Okabe/Kanno Scenario-6 pathology — agents suppress corrective communication because they (falsely) believe alignment exists. The "illusory agreement" CG-BAG identifies as a *structural possibility* becomes, here, a *dynamic event* you can timestamp: the action at which an agent stops querying because its corrupted M″ says the teammate already agrees.

**Why this is your moat:**
- CG-BAG has no recursion and no dynamics — it can say illusory agreement *can exist structurally*, not *when it arises in a process*.
- The ToM-order paper varies recursion *depth*, not the *engine* generating beliefs at each level.
- Your mechanism lives precisely in M″ (the projected layer), corrupted by substrate mismatch, unfolding over the action sequence. Empty cell.

---

## 6. Experimental design (the manipulation table)

Primary 2×2, each cell run 100× per scenario (6 scenarios) per THD (3 values):

| | **Matched substrate** (all same engine) | **Mismatched substrate** (mixed engines) |
|---|---|---|
| **Substrate-aware M″** | Control = Okabe/Kanno replication (when all-BBN) | Gap present, but agents model it correctly |
| **Substrate-naive M″** | Gap absent, naive assumption harmless | **Anthropomorphization Trap cell** — predicted worst coordination |

Plus, for the matched-substrate row, run all three engine types homogeneously (all-BBN, all-Markov, all-tree) to establish each engine's *baseline* TSA curve before any mixing. This isolates "does the engine type alone change dynamics" (a substrate-main-effect) from "does *mismatch* change dynamics" (the interaction you care about).

**Dependent variables (all from Okabe/Kanno, for comparability):**
1. Team accuracy vs. action index (the curve).
2. Actions-to-convergence (efficiency).
3. SD of team accuracy across the 100 runs (stability).
4. **New:** action-index at which corrective communication is suppressed (illusory-agreement onset), where detectable.
5. **New (Translational Friction):** matched-vs-mismatched deltas on (1)–(3).

**Predicted pattern (state as hypotheses, pre-registration-style):**
- H1: Matched-substrate teams reproduce Okabe/Kanno curves (validation, not novelty).
- H2: Mismatched + aware-M″ degrades efficiency/stability modestly (friction cost of translation, even when modeled).
- H3: Mismatched + naive-M″ degrades most, and exhibits earlier/more-frequent illusory-agreement onset (the Trap).
- H4: The substrate-mismatch effect is distinct from, and additive to, observability difficulty (run across all 6 scenarios to show it's not a difficulty artifact).

---

## 7. Domain portability note

The original is a car-fault BBN (Norsys Netica). You have two options:

- **Reuse their domain** (request the BBN/CPTs from Kanno — the paper says to email him). Cleanest for direct curve-comparison, but introduces a dependency and a delay.
- **Build a structurally-equivalent synthetic causal system** (same node count, state cardinalities, edge density, observability split). Faster, fully under your control, and arguably better for a *general* claim about substrate mismatch — but you lose exact curve-overlay with their figures.

Recommendation: build the synthetic system for development and the headline results (cleaner story: "the effect is a property of substrate mismatch, not of cars"), and if time permits, replicate one scenario on their actual domain as a robustness appendix.

---

## 8. Build sequence (do these in order — each gates the next)

**Phase 1 — Replication harness.** Implement the EMBM control logic + Engine A (BBN) only. Reproduce Okabe/Kanno's homogeneous all-BBN curves for at least 2 of their 6 scenarios. **Do not proceed until this matches their published figures within noise.** This is your correctness proof and your reviewers' first sanity check.

**Phase 2 — Engine interface + Engines B, C.** Build the InferenceEngine abstraction, then Markov and tree engines from the shared spec. Validate each *homogeneously* (all-Markov, all-tree teams) — they need not match BBN, but they must be internally sensible (accuracy improves with evidence, converges, etc.).

**Phase 3 — Translation layer.** Implement the `τ` rules. Test on mismatched teams with substrate-*aware* M″ first (the "honest" gap). Measure Friction (H2).

**Phase 4 — Projection manipulation.** Add substrate-naive M″. Run the full 2×2. Hunt for the Trap (H3) and illusory-agreement onset.

**Phase 5 — Sweeps.** All 6 scenarios × 3 THD × 100 runs. Sensitivity analysis as in their Appendix.

A realistic early kill-switch: if Phase 1 won't replicate, the contribution stalls regardless of how good the rest of the design is. Budget for it.

---

## 9. Stack

- **NetLogo** keeps continuity with your Trustv7 work but will fight you on the BBN inference and the tree — you'd be hand-rolling probabilistic inference in a language not built for it.
- **Python** (e.g. `pgmpy` for BBN, `hmmlearn` or hand-rolled for Markov, `scikit-learn` for trees with `predict_proba` as the pseudo-posterior) is the path of least resistance for three heterogeneous engines behind one interface. Recommended unless NetLogo-continuity matters more than build speed.

Whatever the stack: the InferenceEngine interface is the spine. Get it clean and the rest is assembly.

---

## 10. Honest framing reminders for when you write it up

- This is a normative/ideal model (no human error, identical agents except substrate), like the original. The substrate manipulation is the *one* heterogeneity you introduce. Say so.
- Translation rules are modeling commitments, not ground truth. Defend them; ideally show the headline gap survives a charitable τ.
- Position against the three neighbors explicitly: KABOOM (style, not engine), ToM-order (depth, not engine), CG-BAG (structural/static, not dynamic-process; cite as the foundation you extend into dynamics).
- Don't overclaim generality: results are bounded by the diagnosis task and the three engines tested. The original's own limitations section is your template for honesty.

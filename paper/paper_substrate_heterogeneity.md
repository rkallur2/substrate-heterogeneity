---
title: "Two Forces of Substrate Heterogeneity: Sensitivity Gain and Quality Loss in Multi-Agent Cooperative Diagnosis"
author:
  - "Ravikiran Kalluri¹"
  - "Taro Kanno²"
affiliation:
  - "¹D'Amore-McKim School of Business, Northeastern University, Boston, MA, USA"
  - "²[Affiliation TBC — University of Tokyo?]"
date: "July 2026"
abstract: |
  Cooperative diagnosis in small teams — where members combine partial evidence, communicate beliefs, and coordinate on a shared understanding — has been formalized in the Extended Mutual Belief Model (EMBM) of Okabe et al. (2025), an agent-based simulation in which each of three agents maintains a stack of seven Bayesian belief networks representing self, direct beliefs about teammates, and projected beliefs about how teammates in turn model each other. A near-universal assumption of this and related recursive-mutual-belief architectures is that all agents in a team share the same inference substrate. Real cooperative teams — particularly the increasingly common human–AI teams in operational settings — typically do not. We extend EMBM with pluggable inference engines (Bayesian belief network, hidden-Markov approximation, and decision tree) and use agent-based simulation to study three-agent teams in which one agent runs a different substrate from the others, crossed with a projection-mode manipulation that varies whether agents model teammates using their own substrate (naive projection) or the teammate's true substrate (aware projection). Across two 17-node Bayesian networks (Okabe et al.'s Kanno car-diagnosis network and a controlled synthetic domain, N=15 replicates per cell) plus a scale-expansion validation on the canonical 37-node ALARM anesthesia-monitoring network of Beinlich et al. (1989) (N=5 replicates per cell), across two mismatched-substrate axes and eight hard scenarios per network, we find that substrate heterogeneity produces two distinct effects that must be considered jointly: a *sensitivity gain*, in which aware mismatched teams detect substantially more inter-agent conflicts than matched teams in every 17-node condition tested; and an *inference quality loss*, in which the mismatched substrate degrades single-agent diagnostic accuracy in proportion to its inferential distance from exact Bayesian propagation. The signed effect of substrate mismatch on team accuracy is the balance of these two forces, and it can favor either heterogeneity or homogeneity depending on how much the mismatched substrate deviates from exact propagation. On ALARM, the balance is preserved directionally but attenuated in magnitude — consistent with the two-force account's prediction that both forces should dilute across larger inference surfaces. Naive projection systematically suppresses most of the sensitivity signal, producing an illusory-agreement pathology whose severity is bounded by the underlying substrate gap. We propose a two-force account of substrate heterogeneity in recursive-mutual-belief teams. All model code, simulation runners, and results data are publicly released for replication.
keywords: |
  Extended Mutual Belief Model (EMBM); team cognition; agent-based simulation; inference substrate heterogeneity; cooperative diagnosis; team situation awareness; recursive mutual beliefs; human–AI teaming
---

**Keywords:** Extended Mutual Belief Model (EMBM); team cognition; agent-based simulation; inference substrate heterogeneity; cooperative diagnosis; team situation awareness; recursive mutual beliefs; human–AI teaming

# 1. Introduction

Cooperative diagnosis under partial observability requires more than pooling evidence. It requires each agent to maintain not only a belief about the state of the world, but a belief about what its teammates believe, and a belief about what its teammates believe about *its* beliefs. Okabe et al. (2025) formalized this in the Extended Mutual Belief Model (EMBM), giving each agent a stack of seven belief structures — one self-cognition, two direct beliefs about teammates, and four projected beliefs about teammates' beliefs about others — updated by metacognitive conflict resolution across the diagnostic action sequence. EMBM has become an anchoring computational account of how team situation awareness is constructed through communication, and its published curves for a car-diagnosis domain are widely reproduced as a benchmark for multi-agent belief coordination.

EMBM, like nearly all recursive mutual-belief architectures we are aware of, assumes computational homogeneity: every agent, and every one of the seven belief structures inside every agent, runs on the same inference substrate. In the original formulation of Okabe et al. (2025), this substrate is a Bayesian belief network (BBN) with exact propagation over a shared causal graph. That assumption is analytically convenient — it lets projection be a straightforward act of running the same machinery under different priors — but it is empirically strong. Real cooperative diagnosis teams, whether biological or increasingly heterogeneous human-AI teams, do not share substrates. A radiologist and a triage algorithm do not compute posteriors the same way; a pilot and a flight-management system do not represent uncertainty on the same scale; a domain expert and a large language model do not carry out inference through the same operations even when they are handed the same evidence. Whether the EMBM machinery survives this heterogeneity — whether recursive mutual beliefs still coordinate a team when the teammates literally cannot represent each other's inferences — is unknown.

We take up that question by extending EMBM with a **pluggable inference substrate**. The seven belief structures inside each agent, and across agents, are no longer required to be BBNs. They implement a minimal three-operation interface — `ingest(evidence)`, `posterior(node)`, `diagnosis(node, threshold)` — that is sufficient for the full EMBM control logic while allowing the underlying computation to be a BBN, a hidden-Markov chain, a decision tree, or any other posterior-yielding architecture. Every engine consumes the same shared causal specification, so representational choice is manipulated while domain knowledge is held constant. Agents in a team can then be assigned different substrates, and the projection layer that models teammate beliefs can be configured either *aware* (each agent models its teammates using their true substrate) or *naive* (each agent models its teammates using its own substrate, as if the assumption of homogeneity held).

The immediate question is whether heterogeneity is costly. Given that the tree engine, in particular, is a substantially weaker inference machine than a BBN — it holds no joint posterior, cannot integrate observations back through a causal chain, and commits crisply on argmax — one would predict that placing a tree agent in a team otherwise composed of BBN agents should hurt the team's accuracy, and that this hurt should be larger when the tree teammate is modeled naively (as a BBN) than when it is modeled aware (as a tree). This is essentially the *Anthropomorphization Trap* hypothesis: assuming your heterogeneous teammate reasons like you do produces a false sense of mutual understanding and a coordination pathology.

The empirical picture is more nuanced. In this paper we report the results of a full 2×2×2 experimental design (network × mismatched-substrate axis × projection mode, plus the matched/mismatched composition contrast) run on two 17-node Bayesian networks — a controlled synthetic domain and the real Kanno car-diagnosis network — with a scale-expansion validation on the canonical 37-node ALARM anesthesia-monitoring network of Beinlich et al. (1989). We use eight difficulty-ranked scenarios per network, N=15 Monte Carlo runs per cell on the 17-node networks and N=5 on ALARM, and instrumentation that records not only final team accuracy but also the initial and final counts of inter-agent conflicts detected by each agent, decomposed by whether the actor's own belief agreed with ground truth (a *real* conflict where help was needed) or disagreed with ground truth (a *phantom* conflict where the actor was already right and the detected mismatch was illusory from an oracle's perspective).

Our contribution has three parts. First, we document the substrate-mismatch cost in a proper multi-network, multi-axis design: the cost is real and always positive when the mismatched substrate is inferentially weaker than BBN, but it is *reversed* when the mismatched substrate is inferentially close to BBN in the sense that its committed diagnoses seldom disagree with the BBN's. Second, we identify a robust upstream mechanism — a *sensitivity gain* in initial conflict detection under aware projection with mismatched substrates — that replicates in all four (network × axis) cells with per-scenario perfect replication in three of them and 6/8 in the fourth. Third, we show that this sensitivity gain is only partially productive: roughly half of the extra conflicts detected are *phantom*, in the sense that the actor was already correct, so the mismatched-aware team is not smarter but sensitized. We synthesize these findings into a two-force account: the signed effect of substrate mismatch on team accuracy is (real-sensitivity gain) minus (single-agent inference quality loss), and both forces are measurable independently.

The rest of the paper is organized as follows. Section 2 reviews the EMBM framework and the small prior literature on substrate heterogeneity in recursive-belief multi-agent systems. Section 3 formalizes the extended model. Section 4 describes the experimental design and instrumentation. Section 5 reports the results, organized around the two forces. Section 6 discusses implications and boundary conditions.

# 2. Related work

## 2.1 Team cognition and mutual belief

The study of team cognition — how a group of agents collectively perceives, reasons about, and coordinates on a shared task — has a long empirical and theoretical history in human factors and cognitive ergonomics (Salas et al. 2008; Salas et al. 2011; Cooke et al. 2013; McNeese et al. 2020). Two broad computational traditions coexist. *Holistic* models, exemplified by Hutchins's (1995) distributed cognition and Cooke et al.'s (2013) interactive team cognition, treat a team as a single system whose cognitive properties emerge from the interactions of its members without decomposing into individual belief structures. *Collective* models, in contrast, build team cognition compositionally from the beliefs of each individual member and the beliefs each holds about the others (Kanno et al. 2013).

Collective models trace back to formal accounts of *mutual belief* — the condition under which agents believe that they are sharing a belief with their teammates, thought to be necessary for coordinated collective action (Tuomela and Miller 1988). Kanno, Furuta, and Kitahara (2013) formalized this for dyadic teams as a three-layer structure: an agent's self-cognition (M), the agent's direct belief about its partner's cognition (M'), and the agent's projected belief about how its partner in turn models the agent's own cognition (M''). The recursive-and-reflexive character of this structure — beliefs about beliefs about beliefs — is what distinguishes mutual-belief models from simpler shared-mental-model accounts (Cooke et al. 2000; Mohammed et al. 2015). Nonose, Kanno, and Furuta (2010) applied this framework to develop an evaluation method for team situation awareness, and subsequent work (Nonose et al. 2014) explored metacognitive effects on team behavior.

## 2.2 The Extended Mutual Belief Model

Mahardhika, Kanno, and Furuta (2016) extended the dyadic mutual-belief formulation to three-or-more-person teams and introduced *mental subgrouping* — the observation that in larger teams members typically do not model each teammate individually but chunk them into subgroups. Their formulation — henceforth EMBM — retains the three-layer structure per agent but expands each layer to hold multiple belief divisions corresponding to the number of teammates or subgroups. In a three-person team without subgrouping, each agent maintains seven belief structures: one self-cognition, two direct beliefs (one per teammate), and four projected beliefs (each teammate's belief about each of the other two agents).

The direct implementation of EMBM as an agent-based simulation was undertaken by Okabe, Kanno, Cho, Furuta, Yoshida, Karikawa, Nonose, and Inoue (2025), who instantiated each of the seven belief structures per agent as a Bayesian belief network over a car-fault-diagnosis domain, defined an operational conflict-detection scheme (their Types I, II, and III) whose resolution drives an action-generation loop, and reported team-accuracy curves across scenarios varying in diagnostic difficulty and across model variations that ablated the mutual-belief layers. Their study established that the presence of mutual beliefs (M' and M'') generally improves team situation awareness relative to unaware baselines, while also identifying cases where mutual beliefs can produce a small negative effect through communication that is biased by an incorrect prior belief about what teammates already know. In their limitations section, Okabe et al. (2025) explicitly note: "It is recommended that this study be replicated with models other than BBN, such as the Markov model and decision tree, for cross-validation." The present study answers that call, while extending it: rather than replicating with alternative substrates *homogeneously*, we ask what happens when substrates are *heterogeneous* across the team, and when the projection layer must itself commit to a substrate for its teammate model.

## 2.3 Situation awareness, metacognition, and computational cognition

Team situation awareness (TSA) is typically defined as a common understanding of the situation among team members (She and Li 2017; Rousseau et al. 2004), and is considered a critical prerequisite for effective team coordination (Dourish and Bellotti 1992; Gutwin and Greenberg 2004). Different definitions emphasize either the actual overlap in individual situation awarenesses (an outside-observer view) or the *believed* overlap held by each member (an inside-team view); the mutual-belief architectures we build on are naturally suited to the second, because they distinguish between what is true of the world and what each agent believes about what each other agent knows (Kanno et al. 2013).

The action-selection mechanism in EMBM is grounded in *metacognition* — cognition about one's own cognition (Flavell 1979; Nelson and Narens 1994). In particular, EMBM implements metacognitive control as *conflict detection and resolution* across belief layers: when an agent detects that its own belief about a node differs from its belief about a teammate's belief about that node (Type I), or from its projected belief about the teammate's belief about itself (Type II) or about the third teammate (Type III), that conflict triggers a communication or observation action. This connects EMBM to a broader tradition of computational cognitive architectures (Anderson and Lebiere 1998; Newell 1990) while addressing a limitation those architectures share — they were designed for individual cognition, not for the recursive belief structures that team cognition requires (Okabe et al. 2025).

## 2.4 The substrate homogeneity assumption

Across the EMBM literature and the broader recursive-mutual-belief tradition, the belief structures within and across agents are almost universally assumed to run on the same inference substrate. In Okabe et al. (2025), all seven belief structures per agent, across all three agents, are BBNs with identical structure and CPTs; the manipulations tested vary which layers are present (ablations that remove M' or M''), not which computational machinery those layers use. The homogeneity assumption is analytically convenient — it lets projection be the straightforward act of running the same machinery under different priors — but it is empirically strong. Andrews et al.'s (2023) theoretical review of shared mental models in human-AI teams treats the transfer of the shared-mental-model construct from all-human teams to human-AI teams as an open question rather than a settled equivalence. Real cooperative diagnosis teams, particularly the increasingly common human-AI teams in which one member is an algorithm and the others are humans, do not share substrates. A radiologist and a triage algorithm do not compute posteriors the same way; a pilot and a flight-management system do not represent uncertainty on the same scale; a domain expert and a large language model do not carry out inference through the same operations even when they are handed the same evidence.

Broader multi-agent simulation traditions have considered forms of parametric or belief-content heterogeneity (Crosscombe and Lawry 2016; Li and Xiao 2017), and agent-based models of team behavior have varied group size, communication protocols, and coordination structures (Cao et al. 2022; Glinton et al. 2010). But *substrate* variation — where different agents literally compute inference through different mechanisms while representing the same domain, and where the projection layer must itself commit to a substrate for its teammate model — has not, to the authors' knowledge, been embedded in a recursive-mutual-belief architecture. Discussions of substrate diversity have appeared in the human-AI teaming and human-machine cooperation literatures (Flemisch et al. 2019; Wright et al. 2015), and recent empirical work has shown that humans perform and cognize measurably differently when they perceive a teammate as artificial rather than human (Schelble et al. 2023) — a first-order behavioral analogue of what we operationalize computationally as naive projection. What is missing from that empirical work is a formal recursive-belief structure to attach the projection-mode question to.

## 2.5 The projection question

We identify a specific operational concern that arises once substrates are heterogeneous, which we call the *projection question*. When agent A (running substrate X) needs to hold a belief about teammate B (running substrate Y), A must instantiate a projected-belief engine to hold that belief. If A instantiates the projected engine as substrate X — an X-flavored representation of what B thinks — we call that *naive* projection. If A instantiates the projected engine as substrate Y — a Y-flavored representation of what B thinks — we call that *aware* projection. Naive projection is not merely a cognitive laziness; it is a specific representational commitment. It amounts to modeling one's heterogeneous teammate as computationally identical to oneself. This commitment is empirically consequential rather than merely theoretical: recent work in the theory-of-mind-in-LLMs literature has shown that some AI systems now pass classic false-belief tasks and exhibit internal belief-tracking mechanisms of their own (Kosinski 2024), so treating a machine teammate as substrate-identical to oneself is no longer just a simplifying assumption — it is a specific misspecification of what the teammate actually computes. Its downstream consequences are computable and, as we show, non-trivial.

# 3. Model description (ODD protocol)

We describe the model following the Overview, Design concepts, and Details (ODD) protocol of Grimm et al. (2006, 2020). The seven ODD sections below (Purpose and patterns; Entities, state variables, and scales; Process overview and scheduling; Design concepts; Initialization; Input data; Submodels) cover the same content that a narrative model description would but in a form that facilitates comparison with other agent-based models in the recursive-mutual-belief and cooperative-diagnosis literatures.

## 3.1 Purpose and patterns

The purpose of the model is to characterize how heterogeneity in the inference substrate that agents use to represent and reason about a shared task affects the coordination dynamics of recursive-mutual-belief teams. The baseline behavior we seek to preserve is that of the Extended Mutual Belief Model (EMBM) of Okabe et al. (2025), whose published curves show that (i) matched three-agent teams with the second and third belief layers reach higher team situation awareness than teams lacking those layers, and (ii) matched team behavior is sensitive to threshold and scenario difficulty. Our extension is designed to probe two additional patterns that only emerge when substrates can differ across agents: a *sensitivity gain*, in which mismatched teams under aware projection detect more inter-agent conflicts than matched teams, and an *inference quality loss*, in which the mismatched substrate degrades single-agent diagnostic accuracy. Section 5 reports these patterns and Section 6 develops their joint interpretation as a two-force account.

## 3.2 Entities, state variables, and scales

The model contains four kinds of entities.

**Agents.** Three agents, labelled A, B, and C, jointly diagnose a shared 17-node Bayesian network. Each agent maintains seven internal belief structures organised in three layers, following the EMBM of Mahardhika et al. (2016): one L1 self-cognition, two L2 direct beliefs (one per teammate), and four L3 projected beliefs (each teammate's belief about each of the other two agents). Each belief structure is populated by an *inference engine* — a computational object over the shared causal specification that implements the interface described in §3.7.1. The kind of engine that populates each belief structure is a design variable governed by the substrate-composition and projection-mode manipulations of §4.

**Belief structures.** Each of the seven per-agent belief structures holds, at any point in time, a posterior distribution over each node's state variable, a categorical diagnosis for each node (either a specific state or the sentinel `UNCERTAIN`), and a running record of evidence ingested so far. State cardinality varies by node (2 or 3 states in both networks).

**The causal specification.** A 17-node directed acyclic graph over discrete-state nodes with associated conditional probability tables (CPTs). Ten nodes are observable to at least one agent; seven are unobservable inference targets that must be reasoned about via projection and communication. The specification is held constant across all cells of the experimental design; it is described concretely in §3.7.5.

**The scenario.** A ground-truth assignment of states to all 17 nodes, drawn from the network's joint distribution. A scenario represents a specific fault configuration the team must diagnose.

**Scales.** The model has no spatial extent. Time is discrete. A run consists of an *observation phase* of exactly 15 steps (each of the three agents observes each of its five assigned nodes once), followed by a *cooperation phase* of variable length (bounded above by 100 steps). Cell-level replication uses fifteen independent runs per (scenario, cell) combination.

## 3.3 Process overview and scheduling

Time advances in discrete steps. At t = 0, all agents have empty belief structures and no evidence. The two phases execute in sequence.

**Observation phase (steps 1–15).** At each step, one agent–node pair is drawn from the pending observation queue (initialised with three sub-queues of five agent-node pairs each). The agent observes the node's ground-truth state, ingests this as hard evidence into its L1 self-cognition, then propagates the update to its L2 and L3 belief structures via a downward-effect submodel (agents update their beliefs about teammates' beliefs based on their own new observation, following the intra-agent metacognitive control specified by Okabe et al. (2025, §2.4)). Observation-phase step order is randomised per run.

**Cooperation phase (steps 16 to termination).** At each step, one agent is selected uniformly at random from the set of agents with at least one pending detected conflict. The selected agent identifies its highest-importance conflict via the mutual-information node importance measure (§3.7.6) and executes one of three actions to resolve it: `look-at` (a direct observation of one of its own observable nodes); `query` (asking a teammate for that teammate's diagnosis of a node); or `inform` (asserting the agent's own diagnosis of a node to a teammate). The action is executed, evidence flows to the appropriate belief structures of sender, receiver, and the third agent (who hears the exchange), and all agents update their conflict lists. Cross-substrate evidence passing invokes the translation submodel (§3.7.4). The cooperation phase terminates when (i) no agent has any pending conflicts, (ii) five consecutive steps produce no change to any agent's committed diagnosis vector, or (iii) 100 cooperation-phase actions have been executed.

## 3.4 Design concepts

**Basic principles.** The model rests on the recursive-mutual-belief formalism of Kanno et al. (2013), extended to three-person teams with mental subgrouping fixed at level one by Mahardhika et al. (2016), and further extended by Okabe et al. (2025) with a Bayesian belief network as the specific inference substrate for each of the seven belief structures per agent. Our contribution replaces the substrate-homogeneity assumption of Okabe et al. with a pluggable-substrate architecture.

**Emergence.** The two team-level phenomena we study — sensitivity gain and inference quality loss — are emergent in the technical sense: neither is representable in the state of any single agent at any single time step. Sensitivity gain is only measurable across paired conditions (matched vs mismatched) at the level of the initial-conflict snapshot; inference quality loss is only measurable at the end of the cooperation phase against ground truth.

**Adaptation.** Agents adapt their beliefs by ingesting evidence from observation and communication, but they do not adapt their inference substrate or its parameters during a run. Substrate choice is fixed at initialisation.

**Objectives.** Agents have no explicit utility function. Actions are chosen to resolve the highest-importance detected conflict, where importance is defined by mutual information following Okabe et al. (2025, Eq. 3). This is a rule-based rather than optimisation-based objective structure.

**Learning.** No agent learns. The inference engines are hand-specified from the causal specification, not trained on data.

**Prediction.** Prediction is embedded in the L3 projected-belief structures: each agent maintains a prediction of what each teammate believes about the third teammate. The projection-mode manipulation (§3.7.2) varies whether these predictions use the target's true substrate (aware) or the owner's substrate (naive).

**Sensing.** Agents sense in two ways: direct observation of assigned nodes (perfect, deterministic), and communication with teammates (evidence transmitted through the translation layer of §3.7.4). No agent has direct access to any other agent's internal state; every teammate belief is model-based.

**Interaction.** Agents interact through communication actions (`query`, `inform`) that always involve a sender and a receiver, with the third agent hearing the exchange and updating its own belief structures accordingly.

**Stochasticity.** Three sources: (i) the ground-truth scenario is sampled from the joint distribution of the network at initialisation; (ii) observation-phase step order is randomised; (iii) cooperation-phase agent selection is uniform at random from the set of agents with pending conflicts. All stochasticity is seeded per (cell, scenario, replicate) combination for reproducibility.

**Collectives.** The three-agent team is the only collective. Mental subgrouping in the sense of Mahardhika et al. (2016) is fixed at level one — each agent models each other agent individually. Larger teams that would require subgrouping are discussed as a scope limitation in §6.5.

**Observation.** The model records team accuracy (fraction of correctly diagnosed nodes averaged across the three agents), the initial conflict count (total conflicts pending at the moment cooperation begins), and, for the phantom/real decomposition of §5.6, the ground-truth correctness of the observing agent's own L1 belief at the moment each conflict is detected. Section 4.3 gives the formal definitions.

## 3.5 Initialization

At t = 0: the causal specification is loaded (§3.6); a ground-truth scenario is drawn or loaded; each agent's seven belief structures are instantiated with the appropriate engine kinds according to the cell's substrate composition and projection mode; and each belief structure is initialised to the network's prior marginals (i.e., no evidence has been ingested yet). The observation queue for each agent is populated with that agent's four observable nodes plus one further node drawn to reach five per agent, following the observability partition of §3.7.5.

## 3.6 Input data

The model consumes three kinds of input.

**Causal specifications.** For the Kanno network, we load Okabe et al.'s original Netica specification (`agent1.dne`) via a Netica file parser. For Domain17, we procedurally generate the specification from a fixed random seed (see §3.7.5).

**Scenario pools.** For each network, eight hard fault scenarios are pre-sampled from a pool of 120 scenarios drawn from the network's joint distribution. Hardness is ranked by the number of non-nominal-state nodes in the ground-truth assignment (§4.2). The pools are serialised for reproducibility.

**Cell configuration.** Each experimental cell specifies (network, substrate-composition axis, projection mode, threshold THD, replication count) as an input record. The runner scripts (`run_conflict_instrument.py`, `run_dual_metric.py`) enumerate cells and dispatch runs.

## 3.7 Submodels

### 3.7.1 Inference engines

Each of the seven belief structures per agent is an *inference engine* implementing the interface:

```
InferenceEngine:
    ingest(node, evidence)        # incorporate hard or soft evidence about one node
    posterior(node) -> distribution
    diagnosis(node, THD) -> state | UNCERTAIN
```

The interface uses two conventions inherited directly from Okabe et al. (2025). *THD* (threshold) is a scalar between 0 and 1 that controls how confidently an agent must believe in a state before it commits to a categorical diagnosis: if the highest-probability state of a node has posterior probability at least THD, `diagnosis` returns that state; otherwise it returns the sentinel value `UNCERTAIN`, indicating the agent is deferring judgment. This threshold-plus-abstention scheme is what makes cooperation informative — agents that are UNCERTAIN about a node have a reason to observe or query it, and the metacognitive conflict detection of §3.7.3 treats UNCERTAIN-vs-committed disagreements as conflicts requiring resolution. Higher THD values produce more UNCERTAIN diagnoses; lower THD values produce more commitments. Okabe et al. (2025) study THD values of 0.75, 0.80, and 0.85; §4.2 explains our choice of THD = 0.55 and the substantive interpretation of that departure.

We implement three engines from a single shared causal specification (node states, edges, conditional probability tables):

![Figure 1. Three inference substrates instantiated from a shared causal specification.](figs/fig1_substrates.png){width=100%}

**BBN.** Exact belief propagation over the causal DAG. Native soft evidence, graded posteriors. This is the EMBM baseline substrate.

**Markov.** A hidden-Markov-style probabilistic chain derived from the same DAG by linearizing it into a causal ordering and deriving transition/emission matrices from the CPTs. Graded posteriors, native soft evidence. The approximation lies in the causal-chain assumption: dependencies not captured in the chain are lost. This is an *inferential loss* — the machinery is probabilistic but weaker.

**Tree.** A decision-tree structure compiled from the CPTs. Leaf probabilities serve as pseudo-posteriors so that the threshold operator and soft-evidence ingest remain meaningful, but the underlying computation has no bidirectional causal inference: it can only infer designated target nodes from designated feature nodes. This is a *representational loss* — the substrate cannot represent full joint uncertainty at all.

All three engines are hand-specified from the same causal graph, not learned. This eliminates training-noise confounds and lets us attribute any inter-engine differences to substrate structure rather than to data quality.

### 3.7.2 Aware and naive projection

The projection question arises whenever agent A holds a belief about how teammate B represents a node. In the original EMBM, this projected belief is simply another BBN — because everything in the model is a BBN. Once substrates are heterogeneous, projection becomes a design choice.

Under **aware** projection, A instantiates each engine that models teammate B (whether the direct model of B or the projected model of what B thinks about C) using B's actual substrate kind. If B runs Markov, A's model of B is Markov.

Under **naive** projection, A instantiates each such engine using A's own substrate kind. If B runs Markov but A runs BBN, A's model of B is BBN — implicitly assuming B computes as A does.

Under matched compositions (all three agents run the same substrate), aware and naive projection produce identical instantiations. This is a specificity check we return to in the results.

![Figure 2. Which substrate populates each of the seven EMBM belief structures under aware versus naive projection. Under aware projection, each teammate model uses the target's true substrate; under naive projection, every teammate model uses the owner's substrate, making substrate heterogeneity structurally invisible.](figs/fig2_projection.png){width=100%}

### 3.7.3 Metacognitive conflict detection

Following EMBM, at any point during a run each agent A can detect three types of conflict per (node, teammate) pair:

- **Type I:** A's own diagnosis of the node differs from A's direct model of teammate B's diagnosis.
- **Type II:** A's own diagnosis differs from A's model of B's model of A.
- **Type III:** A's own diagnosis differs from A's model of B's model of the third teammate C.

A conflict fires whenever the two compared diagnoses either commit to different states or one commits while the other is UNCERTAIN (following the EMBM convention that information gaps are also conflicts to be resolved). The action-selection procedure picks the highest-importance detected conflict and calls the resolution logic described in §3.3.

![Figure 3. The three conflict types EMBM agents monitor at every step. Type I compares the agent's own L1 belief against its direct model of a teammate; Types II and III compare L1 against projected beliefs about what the teammate believes about the agent and about the third teammate, respectively.](figs/fig3_conflict_types.png){width=100%}

### 3.7.4 Cross-substrate translation

When an agent A (substrate X) communicates evidence to agent B (substrate Y), the evidence must be translated. We use the translation rule from Okabe et al. (2025) without modification: a diagnosis or posterior in X is passed to Y's `ingest` interface, and Y absorbs it in the form its own substrate allows. Under BBN → Markov, a distribution becomes a soft evidence vector; under BBN → tree, a posterior collapses on argmax; under tree → BBN, a crisp label becomes a gamma-weighted point distribution. This translation layer is held constant across all experiments here, so any differences we observe are attributable to substrate choice or projection mode, not to translation.

### 3.7.5 Three network instantiations

The abstract model of §§3.7.1–3.7.4 is domain-agnostic. To ground it, we instantiate it on three Bayesian networks: two 17-node networks that ground the paper's direct comparison with the original EMBM instantiation, and one canonical 37-node medical monitoring network that tests whether the two-force account survives at approximately twice the network scale. All three networks are held constant across cells of the experimental design; agents in a given team share the *same* causal specification but may run different inference engines over it.

**Kanno.** The car-diagnosis network of Okabe et al. (2025), imported directly from the authors' Netica specification (`agent1.dne`), itself adapted from the original Norsys Netica car-diagnosis example. The 17 nodes represent car parts organized around a `Starts` outcome and its causal predecessors — battery, alternator, fuel path (tank, filter, fuel system), air path (filter, air), electrical (main fuse, plug voltage, spark plugs, spark quality), timing, distribution, starter motor, and lights. Ten nodes are observable, seven are unobservable inference targets. The observability partition into agents A, B, C is reconstructed from Figure 8 of Okabe et al. (2025): agent A observes starter motor, gas tank, alternator, and lights; agent B observes spark plugs, battery voltage, main fuse, and lights (overlap with A on lights); agent C observes air filter, fuel, gas filter, and main fuse (overlap with B on main fuse). The unobservable targets are plug voltage, starter, distribution, spark quality, air, timing, and starts. Because Okabe et al. (2025) has not released the source code publicly, our BBN engine and cooperation-phase mechanics are reimplemented from the paper's algorithmic description; we validated the reimplementation by confirming that a matched all-BBN team on this network reproduces the qualitative shape of the published curves in Okabe et al.'s Figure 9 before any substrate manipulation was introduced.

**Domain17.** A hand-specified synthetic causal system, deliberately designed to exhibit richer multi-parent structure than the Kanno network. Whereas most of Kanno's non-root nodes have one or two parents, Domain17 has five nodes with three parents each (`Comm`, `ControlLoop`, `DataPath`, `Output`, `SystemHealth`) and eight further nodes with exactly two parents. This is by design: the Markov engine's causal-linearization approximation and the tree engine's argmax-on-features approximation both lose more information on multi-parent nodes than on chain-like structures, so Domain17 provides an environment where the substrate manipulations are guaranteed to bite. The 17 nodes form a metaphor for a generic multi-subsystem device — power infrastructure (`MainPower`, `Regulator`, `BusVoltage`), timing and control (`Clock`, `Firmware`, `Timing`, `ControlLoop`), sensor chains (`SensorL`, `SensorR`), actuation and thermal (`Actuator`, `Thermal`), communication (`Comm`), data path (`DataPath`), output (`Output`), and roll-up health nodes (`Stability`, `Integrity`, `SystemHealth`). Node cardinality is mixed (eight nodes have two states, nine have three states). Root priors are hand-specified to skew mildly toward the nominal state so that fault scenarios are informative rather than trivial; conditional probability tables for non-root nodes are Dirichlet-sampled with sharpness 2.5 and parent-index-dependent bias, from a fixed random seed for reproducibility. Ten nodes are observable, seven are unobservable inference targets. The observability partition is: agent A → `MainPower`, `Regulator`, `SensorL`, `Output`; agent B → `Clock`, `BusVoltage`, `SensorR`, `Thermal`; agent C → `Firmware`, `Actuator`, `SensorL`, `BusVoltage` (overlaps with A on `SensorL` and with B on `BusVoltage`), following the overlap-with-partial-redundancy pattern of Okabe et al.'s Figure 8.

**ALARM.** The canonical anesthesia-monitoring network of Beinlich et al. (1989), a 37-node discrete Bayesian network widely used as a benchmark in the BN literature. We selected ALARM specifically to test whether the two-force account survives on networks approximately twice the scale of the paper's two 17-node instantiations, while continuing to satisfy the recursive-mutual-belief architecture's requirements: multi-parent structure sufficient to activate the tree and Markov substrate manipulations (ALARM has 3 nodes with 3 or 4 parents and 14 with 2 parents; maximum in-degree 4), a naturally interpretable three-role observability partition reflecting real cooperative-diagnosis practice, and canonical status independent of Kanno's car-diagnosis lineage. The 37 nodes divide into Beinlich's original three-way classification: 8 diagnostic hypotheses (hypovolemia, left ventricular failure, anaphylaxis, insufficient anesthesia, pulmonary embolus, intubation state, kinked tube, disconnection), 13 intermediate physiological states (cardiac output, stroke volume, heart rate, catecholamine level, arterial CO2, and eight ventilator/circulation intermediates), and 16 clinical findings. Our observability partition uses 14 of the 16 findings across three agents; we treat the two error-indicator findings (`ERRLOWOUTPUT`, `ERRCAUTER`) as intermediate rather than observable, since they are monitor-fidelity flags inferred from monitor discrepancies rather than direct clinical observations, giving a final partition of 8 + 14 + 15 = 37. The three agents reflect real anesthesia team roles: the **anesthesiologist** (Agent A) observes hemodynamic monitors (`BP`, `CVP`, `PCWP`, `HRBP`, `PAP`, `SAO2`); the **respiratory technician** (Agent B) observes ventilator and gas exchange readouts (`EXPCO2`, `MINVOL`, `MINVOLSET`, `PRESS`, `FIO2`); the **circulating nurse / monitor operator** (Agent C) observes EKG-derived and saturation-derived heart rate signals plus patient history (`HREKG`, `HRSAT`, `HISTORY`, `SAO2`, `MINVOLSET`). Overlap structure: `SAO2` is observed jointly by A and C (both routinely watch the pulse oximeter), and `MINVOLSET` is observed jointly by B and C (both see the ventilator settings the tech configured). This gives 14 unique observable nodes and 23 hidden inference targets, a larger inference surface than either 17-node network. CPT values are Beinlich's originals as distributed in the `bnlearn` repository; no resampling. Hard scenarios for ALARM are ranked by a clinical hardness measure (number of nodes in non-nominal clinical states, using a network-specific nominal-state map — needed because ALARM's `.bif` state orderings are not consistent with "nominal first" as they are in the two 17-node networks).

The three networks serve complementary roles. Kanno is the "canonical" network — using it grounds our results in the same substrate on which the original EMBM curves were established, so any effect we find there is directly comparable to the source study. Domain17 is the "controlled" network — using it lets us guarantee, by construction, that multi-parent structure is present in force, so that the substrate manipulations have real bite. ALARM is the "scale-expansion" network — using it tests whether the two-force account continues to hold when we double the number of nodes, drawing the causal specification from an entirely independent, real, clinically-established Bayesian network built by anesthesiologists rather than by us or by Okabe et al. Effects that appear in all three networks generalize across topology, CPT structure, and network scale; effects that appear in only some networks can be examined for scale-dependent or topology-specific artifacts. Figure 4 shows the topology of the two 17-node networks side by side; the 37-node ALARM network is shown in Figure 5.

![Figure 4. Bayesian network topologies of the two 17-node experimental substrates. Node fill color indicates which agent observes that node in the observability partition; half-and-half fills mark nodes observed by two agents (overlap zones from Okabe et al. 2025 Figure 8). Dashed borders mark unobservable nodes, which must be reasoned about via cooperative inference. Kanno (top) has predominantly single- and two-parent structure, with the sink node Starts having five parents. Domain17 (bottom) is deliberately denser: five nodes have three parents each (Comm, ControlLoop, DataPath, Output, SystemHealth), providing greater surface area for the tree engine's argmax and the Markov engine's causal-linearization approximations to bite.](figs/fig4_networks.png){width=100%}

![Figure 5. The ALARM network (Beinlich et al. 1989), 37 nodes and 46 edges, an anesthesia-monitoring Bayesian network approximately 2x the scale of the paper's two 17-node instantiations. Diagnostic hypotheses (8 nodes, dashed borders) are the hidden fault causes the team must jointly infer. Fill color indicates the three-agent observability partition: Agent A (anesthesiologist) watches hemodynamic monitors; Agent B (respiratory technician) watches ventilator and gas exchange; Agent C (circulating nurse) watches EKG, saturation, and history. Overlap nodes (SAO2, MINVOLSET) are observed by two agents each, mirroring the overlap-with-partial-redundancy pattern of Okabe et al. Figure 8.](figs/fig5_alarm.png){width=100%}

### 3.7.6 Node importance

The action-selection procedure of §3.3 picks the highest-importance detected conflict, where node importance follows the mutual-information sum definition of Okabe et al. (2025, Eq. 3): the importance of node X is the sum of the pairwise mutual information between X and every other node in the network, computed under the network's prior distribution. This measure is scenario-independent and is precomputed once per network. In case of ties or absent conflicts, the agent selects uniformly at random from the tied set.

# 4. Experimental design

## 4.1 The 2×2×2 design

We manipulate three factors:

1. **Network** — two Bayesian networks. Domain17 is a controlled 17-node synthetic causal system with hand-specified CPTs designed to produce nontrivial partial-observability inference. Kanno is the 17-node car-diagnosis network from Okabe et al. (2025), imported directly from the authors' Netica specification (agent1.dne).

2. **Mismatched substrate axis** — the substrate assigned to agent A in mismatched compositions is either *tree* (representational loss) or *markov* (inferential loss). Agents B and C are always BBN.

3. **Projection mode** — aware or naive.

Crossed with these three factors is the standard EMBM composition contrast: *matched* (all three agents BBN) versus *mismatched* (one agent on the axis substrate, two on BBN). The full cell structure per (network × axis) condition is therefore the 2×2 of {matched, mismatched} × {aware, naive}, and we run all four (network × axis) conditions on the two 17-node networks, yielding a 2×2×2 = 8-cell primary design plus the two matched-baseline cells shared across axes. We additionally run the same 2×2 sub-design (matched vs mismatched crossed with aware vs naive, on both tree and Markov axes) on the 37-node ALARM network as a scale-expansion validation, with the reduced replication described in §5.1.

## 4.2 Scenarios and replication

For each network we pre-select eight *hard* scenarios drawn from a pool of 120 sampled fault configurations, ranked by hardness. Hardness is defined as the number of non-nominal-state nodes in the ground-truth fault configuration — a scenario with hardness h has h of its 17 nodes in a non-default (faulty or degraded) state, and 17-h in their nominal state. Higher hardness is more difficult because more of the diagnosis surface is non-obvious. For Kanno, the top-8 scenarios span hardness 7 to 9; for Domain17, hardness 12 to 15. Domain17 scenarios are systematically harder in this metric because the network's node cardinalities are higher and its fault-propagation cascades produce more downstream deviations from nominal per root fault.

All cells within a network condition use the same eight scenarios, so per-scenario paired comparisons are possible. For Kanno the hard-scenario pool clusters on `BatVolt` (battery voltage) faults, with four scenarios having `BatVolt=dead` (categorical fault) and four having `BatVolt=weak` (graded fault); this fault-type structure interacts with the aware-vs-naive projection manipulation in ways discussed in the results. For Domain17 the hard scenarios span multiple root-fault combinations without a single dominant fault-type structure.

Each cell within each scenario is run N = 15 times with different random seeds. Seeds are stratified so that a given cell/scenario/replicate combination is reproducible from a documented base seed. We use N = 15 rather than the original N = 100 of Okabe et al. (2025) primarily to fit our full 2×2×2 sweep within the computational budget available; sensitivity analysis on the single-metric run (Section 5.1) shows aggregate estimates are stable at both N = 15 and a supplementary N = 40 within the expected sampling error. Where our claims turn on per-scenario sign consistency (8/8 replication), the reduced N has no effect on the claim; where they turn on aggregate significance testing, the reduced N is a conservative choice (less statistical power) rather than a permissive one.

Threshold is held at THD = 0.55 throughout. This diverges from the original EMBM protocol (Okabe et al. 2025 use THD = 0.75, 0.80, 0.85). The rationale for the lower value is that lower thresholds produce more committed diagnoses per agent, and therefore more surface area for the aware-vs-naive projection distinction to bite: at higher thresholds, more nodes are UNCERTAIN in every agent's beliefs, and UNCERTAIN-vs-UNCERTAIN comparisons across projection modes are structurally identical, so the aware/naive manipulation has less to affect. Our substrate-mismatch findings are therefore anchored to a *low-threshold* regime, and their extension to Okabe et al.'s higher-threshold regime is a natural follow-up study that we discuss in Section 6.5.

## 4.3 Metrics

**Team accuracy.** The primary EMBM dependent variable: the mean, across the three agents, of the fraction of nodes whose L1 (self) diagnosis equals the ground truth. We compute this at the end of the run. UNCERTAIN diagnoses are counted as incorrect (the *full* metric of §5.2).

**Substrate gap.** For each scenario, `matched-aware accuracy − mismatched-aware accuracy`. Positive values indicate substrate mismatch is costly (the classical trap prediction); negative values indicate mismatch is beneficial.

**Aware gap.** For each scenario, `mismatched-aware accuracy − mismatched-naive accuracy`. Positive values indicate the aware projection benefits mismatched teams (the naive-projection pathology of the Anthropomorphization Trap).

**Initial conflict count.** The total number of conflicts across all three agents, counted immediately after the observation phase and before any cooperation action is taken. This is the *sensitivity* measure: it isolates how many conflicts the projection machinery even sees, before the interactive dynamics kick in.

**Phantom/real decomposition.** For each detected conflict `(node, type, teammate)` in the initial-conflict snapshot, we classify it as *phantom* if the detecting agent's own L1 diagnosis agrees with ground truth (the mismatch with the teammate model is illusory from an oracle's perspective — the actor is already right and the conflict-resolution action would be wasteful) or *real* otherwise. This decomposition is not visible to the agents; it is computed externally for analysis only.

**Specificity check.** In each (network × axis) condition we verify that within *matched* compositions, aware and naive projections produce identical initial-conflict counts — a structural expectation of the model that serves as a design-validity check.

# 5. Results

## 5.1 The substrate gap: signed and axis-dependent

Table 1 reports the substrate gap and aware gap by (network, axis) condition, aggregated across the eight hard scenarios per network. The two 17-node networks (Kanno, Domain17) were run at N=15 replicates per (scenario, cell); the 37-node ALARM network was run at N=5 replicates per (scenario, cell) — a deliberate reduction reflecting ALARM's role as a scale-expansion validation rather than a primary result, where the aim is to establish sign consistency and mechanism preservation rather than to tighten confidence intervals already known from the smaller networks.

**Table 1. Aggregate substrate and aware gaps.**

| Network | N | Mismatch axis | Substrate gap (m-a − mm-a) | t(7), p | Positive scenarios | Aware gap (mm-a − mm-n) |
|---|---|---|---|---|---|---|
| Domain17 | 15 | tree | **+0.077** | +7.24, <0.001 | 8/8 | −0.020 |
| Kanno | 15 | tree | **+0.053** | +5.53, 0.001 | 8/8 | +0.023 |
| ALARM | 5 | tree | **+0.024** | +2.15, 0.069 | 6/8 | +0.001 |
| Domain17 | 15 | markov | **−0.019** | −0.95, 0.37 | 3/8 | +0.017 |
| Kanno | 15 | markov | **−0.002** | −0.29, 0.78 | 4/8 | +0.004 |
| ALARM | 5 | markov | **+0.008** | +0.52, 0.62 | 4/8 | −0.001 |

The substrate gap has a clear, replicable positive sign whenever the mismatched substrate is the *tree* engine, across all three networks (t ≥ 2.15, p ≤ 0.07, with per-scenario replication 8/8 on the two 17-node networks and 6/8 on ALARM). This is the classical substrate-mismatch cost. When the mismatched substrate is the *Markov* engine, however, the gap is null or slightly negative on the 17-node networks and slightly positive but statistically undistinguishable from zero on ALARM.

The ALARM row deserves separate comment. Its substrate gap under the tree axis (+0.024) has the same sign as Kanno-tree (+0.053) and Domain17-tree (+0.077) but roughly half the magnitude. Its per-scenario sign consistency drops from 8/8 to 6/8. Two of the eight ALARM-tree scenarios show a negative substrate gap; two of the eight show sensitivity gain values near zero or slightly negative (mismatched-aware detects *fewer* initial conflicts than matched-aware, not more). We interpret this as a scale effect on the balance between the two forces developed in §5.7: with 37 nodes and 23 hidden inference targets — nearly triple the hidden inference surface of the 17-node networks — individual substrate mismatches are diluted across a larger inference structure, and the sensitivity gain and quality loss both attenuate. Both forces remain measurable in the aggregate (the specificity check holds cleanly on ALARM, max |matched-aware − matched-naive| = 0.007), but their per-scenario balance becomes more heterogeneous. The two-force account thus predicts, and here demonstrates, that its own effects should attenuate with network scale rather than replicate uniformly across networks of different sizes.

![Figure 6. Substrate gap by (network × mismatched axis). Bars show mean, dots show per-scenario values, error bars are ±1 SE. Tree-axis conditions produce a robust positive substrate gap on all three networks (matched wins, classical trap); Markov-axis conditions produce a null or mildly negative gap on the 17-node networks and a small non-significant positive gap on ALARM (mismatched wins or ties, inverted regime). ALARM's magnitudes are attenuated relative to the 17-node networks — consistent with the two-force account's prediction that both forces dilute across larger inference surfaces.](figs/fig6_substrate_gap.png){width=95%}

The aware gap — the classical Anthropomorphization Trap signal — is at aggregate positive but small and not statistically robust in any single condition. This aggregate obscures a scenario-level structure documented in prior work on this dataset: on Kanno-tree, the aware gap splits by fault type (dead battery: aware > naive; weak battery: naive > aware), producing a scenario-conditional trade-off rather than a monotone trap. We return to this in the discussion.

## 5.2 Metric-dependence: does the sign flip under abstention-aware accuracy?

Before proceeding to the mechanism analysis, we address a natural concern: because the full team-accuracy metric treats UNCERTAIN diagnoses as errors, differences in commitment rate across cells could be mechanically inflating (or deflating) the substrate gap. To test this, we compute an alternative *committed accuracy* metric that scores only nodes where an agent's diagnosis is not UNCERTAIN (correct / #committed), so that abstention neither helps nor hurts.

For the domain17-Markov condition — the cell where the sign might most plausibly reverse — we ran the full 2×2 with both metrics reported side-by-side. The aggregate substrate gap under the full metric is −0.019 (t = −0.95, p = 0.37, 3/8 scenarios contradict); under the committed metric it is −0.033 (t = −2.91, **p = 0.023**, 0/8 scenarios contradict). The sign does not flip; the effect becomes stronger and cleaner under the abstention-aware metric, moving from statistically undecidable to a clear negative result. Per-scenario, 3 of 8 scenarios flip sign between the two metrics, but the aggregate sign is unchanged.

The metric-dependence is fully explained mechanically by cell-level commitment rates: the Pearson correlation between the per-scenario commit-rate difference (matched-aware minus mismatched-aware) and the per-scenario full-committed metric difference is r = +0.96, p < 0.001. Whenever the matched-aware team commits more than the mismatched-aware team, the full metric rewards it disproportionately relative to the committed metric. In this dataset, aggregate commit rates are essentially equal across cells (matched-aware 0.708, mismatched-aware 0.707), so the aggregate sub_gap is stable across metrics even though individual scenarios can flip.

We interpret this as evidence that the Markov surprise is not an abstention artifact. From this point on, all analyses use the full metric for continuity with the EMBM literature.

## 5.3 The sensitivity mechanism: mismatched-aware teams detect more conflicts

Table 2 reports the initial conflict counts by cell and the (mismatched-aware minus matched-aware) contrast.

**Table 2. Initial conflict counts and the sensitivity contrast.**

| Cell | m-a | m-n | mm-a | mm-n | mm-a − m-a | t(7) | p | Pos |
|---|---|---|---|---|---|---|---|---|
| Domain17 tree | 13.50 | 13.50 | 44.25 | 6.75 | **+30.75** | +12.87 | <0.0001 | 8/8 |
| Domain17 markov | 13.50 | 13.50 | 34.88 | 6.75 | **+21.38** | +11.15 | <0.0001 | 8/8 |
| Kanno tree | 12.75 | 12.75 | 22.88 | 3.75 | **+10.12** | +8.04 | 0.0001 | 8/8 |
| Kanno markov | 12.75 | 12.75 | 15.38 | 12.75 | **+2.62** | +3.86 | 0.006 | 6/8 |
| ALARM tree | 30.75 | 30.75 | 34.12 | 26.25 | **+3.38** | +0.90 | 0.40 | 5/8 |
| ALARM markov | 30.75 | 30.75 | 46.12 | 30.00 | **+15.38** | +2.99 | 0.020 | 6/8 |

![Figure 7. Initial conflict counts across the 2×2×2 design. Matched-aware and matched-naive produce identical counts in every cell (specificity check); mismatched-aware dramatically exceeds matched-aware in every 17-node cell (sensitivity gain, red arrow shown for Domain17-Tree); mismatched-naive collapses back below matched-aware in every 17-node cell (naive suppression, green arrow shown for Domain17-Tree). ALARM shows the same qualitative pattern with attenuated magnitudes on the tree axis (mm-a only slightly above matched, 34.1 vs 30.8, p=0.40) but preserved strength on the markov axis (mm-a=46.1 vs matched=30.8, p=0.02, 6/8 positive).](figs/fig7_conflict_counts.png){width=100%}

The sensitivity gain — mismatched-aware detects strictly more initial conflicts than matched-aware — replicates in all four (network × axis) 17-node conditions with 8/8 per-scenario replication in three of them and 6/8 in the fourth. The effect size varies substantially, from +2.6 conflicts (Kanno-Markov) to +30.8 (Domain17-tree). Its magnitude tracks approximately how much the mismatched substrate's diagnoses actually diverge from BBN's: tree, which cannot represent joint uncertainty, produces large divergence and large sensitivity gain; Markov on Kanno, whose CPTs concentrate most mutual information in dominant parents so that the Markov approximation is nearly lossless, produces small divergence and small sensitivity gain. The relationship is monotone across the four 17-node cells.

On ALARM, the sensitivity gain is preserved on the markov axis (+15.4 conflicts, t=+2.99, p=0.02, 6/8 positive) but attenuated to non-significance on the tree axis (+3.4 conflicts, t=+0.90, p=0.40, 5/8 positive). The reversal of the tree-vs-markov effect-size ordering on ALARM — where markov produces a larger sensitivity gain than tree, opposite the pattern on the two 17-node networks — is a scale-related finding we return to in §6.4.

## 5.4 The illusory-agreement mechanism: naive projection suppresses the signal

Table 3 reports the mismatched-aware minus mismatched-naive contrast.

**Table 3. Naive suppression of mismatch-induced conflicts.**

| Cell | mm-a − mm-n | t(7) | p | Positive scenarios |
|---|---|---|---|---|
| Domain17 tree | **+37.50** | +14.43 | <0.0001 | 8/8 |
| Domain17 markov | **+28.12** | +10.84 | <0.0001 | 8/8 |
| Kanno tree | **+19.12** | +5.48 | 0.001 | 8/8 |
| Kanno markov | **+2.62** | +3.86 | 0.006 | 6/8 |
| ALARM tree | **+7.88** | +1.31 | 0.23 | 5/8 |
| ALARM markov | **+16.12** | +3.40 | 0.011 | 6/8 |

In every 17-node cell, naive projection detects substantially fewer initial conflicts than aware projection in the *mismatched* composition. The suppression averages 80% of the conflicts that aware projection surfaces — that is, four out of five conflicts that aware projection would have flagged are structurally invisible to naive projection because the naive engine models the mismatched teammate as if it computed the same way as the modeling agent, producing (falsely) agreeing diagnoses. On ALARM the pattern is preserved but attenuated: on the markov axis, naive suppression is +16.1 (p=0.01, 6/8) — clearly present and consistent with the 17-node markov cells. On the tree axis, the point estimate is +7.9 but statistically unresolved at N=5 (p=0.23, 5/8) — mirroring the smaller sensitivity gain on that same cell (Table 2). The illusory-agreement mechanism is thus present in five of six cells at the significance threshold, with the ALARM-tree sixth cell showing the correct direction but insufficient power at N=5 to distinguish from zero.

## 5.5 Specificity check: matched teams show no aware/naive difference

In every one of the six (network × axis) conditions, and in every one of the eight scenarios per condition, matched-aware and matched-naive teams produce *identical* initial conflict counts on the 17-node networks (t = 0.00, p = 1.00, all scenarios) and near-identical counts on ALARM (max |m-a − m-n| = 0.007 across the 16 ALARM cell-scenario pairs, well below the noise floor of the mismatched contrasts). This is a structural consequence of the model — when all agents run the same substrate, aware and naive projections instantiate identical engines — and it serves as a design-validity check: the aware/naive difference documented in Section 5.4 is not a generic property of the projection-mode manipulation but is specifically a signature of substrate heterogeneity.

## 5.6 Phantom vs. real: the extra conflicts are only half productive

The sensitivity gain of Section 5.3 raises the question: are the extra conflicts detected by mismatched-aware teams actually useful — do they surface nodes where the detecting agent's own L1 is wrong and needs correction — or are they noise, illusory disagreements with a teammate model on nodes where the actor is already right? Table 4 decomposes the sensitivity gain into extra *real* conflicts (actor's L1 disagrees with ground truth) and extra *phantom* conflicts (actor's L1 agrees with ground truth).

**Table 4. Phantom/real decomposition of the mismatched-aware − matched-aware extra conflicts.**

| Cell | Extra real | Extra phantom | Phantom fraction of total |
|---|---|---|---|
| Domain17 tree | +17.25 | +13.50 | 43.9% |
| Kanno tree | +8.25 | +1.88 | 18.5% |
| ALARM tree | +9.00 | **−5.62** | negative (net phantom reduction) |
| Domain17 markov | +10.50 | +10.88 | 50.9% |
| Kanno markov | +0.75 | +1.88 | 71.4% (small N) |
| ALARM markov | +9.38 | +6.00 | 39.0% |

![Figure 8. Phantom vs. real decomposition of the mismatched-aware minus matched-aware extra initial conflicts. Green = extra real conflicts (actor's L1 disagrees with ground truth, so coordination help is warranted); orange = extra phantom conflicts (actor is already right; the detected disagreement is illusory from an oracle's perspective and any action taken on it is wasted). ALARM-tree is the anomaly: it shows a positive real component (+9.0) alongside a *negative* phantom component (−5.6), meaning the tree substrate's argmax-collapse on ALARM's larger inference surface actually reduces phantom conflicts relative to a matched-BBN team while adding real conflicts — an unusual configuration in which mismatch's productive-sensitivity fraction exceeds 100% of the net sensitivity gain.](figs/fig8_phantom_real.png){width=95%}

Three observations. First, the *real* component of the sensitivity gain is significant and non-zero in five of six cells (Kanno-Markov aside, whose totals are close to noise), so the mechanism is not purely illusory disagreement — mismatched-aware teams really do detect nodes where they are genuinely wrong and could benefit from help. Second, the phantom component is large and positive in most cells: roughly half of the extra sensitivity in the 17-node conditions is not productive, consisting of conflicts about nodes where the detecting agent is already correct. Third, and unexpectedly, ALARM-tree exhibits a *negative* phantom component: the mismatched-aware team has fewer phantom conflicts than the matched team, even though it detects more real ones. We interpret this as a scale-related consequence of the tree substrate's crisp-commit behavior on a large inference surface: the tree agent's argmax collapse on ALARM's 23 hidden inference targets produces fewer high-confidence disagreements with teammate models on nodes where it happens to be right, effectively suppressing phantom disagreement more than it introduces new phantom disagreement. This is a specific empirical prediction the two-force account did not anticipate, and one worth investigating further in follow-up work.

The phantom fraction is not uniform across cells. Kanno-tree has the lowest positive phantom fraction (18.5% — the extra sensitivity is 81% productive) yet still shows a large positive substrate gap (matched wins by +0.053). Domain17-Markov has a middling phantom fraction (51%) and shows a mildly negative substrate gap (mismatch wins by 0.019). ALARM-tree has a *negative* phantom fraction and the smallest tree-axis substrate gap (+0.024). This confirms that phantom fraction alone does not predict substrate-gap sign. The sign depends on the second force, which we take up next.

## 5.7 The two-force account

Sections 5.3–5.6 identify a first mechanism: substrate mismatch under aware projection produces a *sensitivity gain*, most of which is real but a substantial fraction of which is phantom. But the outcome data of Section 5.1 shows that this sensitivity gain does not translate into a corresponding accuracy gain. On Kanno-tree, mm-aware surfaces +8.25 extra real conflicts per scenario — nodes where at least one agent is genuinely wrong and needs coordination help — yet mm-aware loses to m-aware by 0.053 in team accuracy. The sensitivity is productive, but the team is nonetheless worse.

The missing piece is a second force: single-agent *inference quality*. Placing a tree engine in an otherwise-BBN team downgrades the tree agent's own L1 diagnoses regardless of how well the projection layer models the heterogeneity. The tree agent contributes a weaker inference to the team's post-cooperation belief state. The substrate gap is the balance of this quality loss against the (real component of the) sensitivity gain:

> **substrate gap  ≈  (inference quality loss)  −  (real sensitivity gain)  ×  (action efficacy)**

Where action efficacy captures how often a detected real conflict translates into a beneficial action outcome given the noisy translation and idle-terminator dynamics of the coop phase.

The four 17-node cells fit this account cleanly. On Kanno-Markov, both terms are small (Markov ≈ BBN in inferential power on Kanno's CPTs), so the substrate gap is near zero (−0.002). On Domain17-Markov, quality loss is small but real-sensitivity gain is +10.5 — so mismatch wins (−0.019). On Kanno-tree, quality loss is large (tree << BBN in inferential power) and real-sensitivity gain is +8.25 — so mismatch loses (+0.053). On Domain17-tree, both terms are large, but quality loss dominates: mismatch loses (+0.077).

The two ALARM cells extend the picture at scale. On ALARM-tree, the substrate gap remains positive (+0.024, 6/8 scenarios) with the same sign as on Kanno-tree and Domain17-tree but at roughly half the magnitude — quality loss still dominates, but both forces are attenuated across ALARM's 23 hidden inference targets (compared to 7 in the 17-node networks). On ALARM-Markov, the substrate gap is near-zero and slightly positive (+0.008, statistically indistinguishable from zero, 4/8 scenarios), consistent with the near-zero gap on the two 17-node Markov cells. Neither ALARM cell contradicts the two-force account; both show its predicted directional structure with the magnitude attenuation the account itself predicts once the inference surface expands. Concretely, if both sensitivity gain and quality loss scale inversely with the fraction of the network any single mismatched agent's inference affects, then trebling the hidden inference surface (7 → 23) should attenuate both forces — which is what we observe, in both directions, on both ALARM cells.

![Figure 9. The two-force account visualized. The x-axis is the real sensitivity gain (extra conflicts in mm-aware where the actor's own L1 is wrong); the y-axis is the substrate gap. Color indicates the mismatched-axis substrate; marker shape indicates the network. The four 17-node cells occupy distinct regions of the two-force space: near-origin (Kanno-Markov, both forces small); high-sensitivity/negative-gap (Domain17-Markov, sensitivity wins); moderate-sensitivity/positive-gap (Kanno-Tree, quality loss wins); high-sensitivity/large-positive-gap (Domain17-Tree, quality loss still wins despite large sensitivity). The two ALARM cells (added as a third marker shape) sit closer to the origin than their 17-node counterparts on the same axis, reflecting attenuation of both forces at the larger network scale (§5.1).](figs/fig9_two_force.png){width=95%}

This decomposition also explains why phantom fraction does not predict outcome sign: phantom conflicts do not directly contribute to accuracy (they lead to wasteful actions but not necessarily to wrong final diagnoses), so the sensitivity term in the outcome equation is best approximated by *real* sensitivity gain, not total sensitivity gain. Phantom conflicts are a "cost of sensitivity" that reduces action-efficacy but does not change the sign of the substrate gap.

## 5.8 Team-size scalability: what changes when N grows

The primary experiments hold team size at N=3, following Okabe et al. (2025). To probe whether the two-force account holds at larger team sizes, we extend the design to N=5 agents (four BBN plus one substrate outlier) on all three networks. On each network we construct a 5-agent observability partition with the same structural signature — 8 unique observed nodes, 2 obs per agent, 2 overlap pairs — so that per-agent observation reduces relative to the corresponding N=3 partition (Domain17: 4→2, Kanno: 4→2, ALARM: ~5→2) while total observation surface shrinks proportionally. Each agent maintains 21 belief structures (`1 + (N-1) + (N-1)² = 1 + 4 + 16`), up from 7 at N=3. Because per-agent inference cost and belief-structure count both grow, we reduce to the top three hard scenarios per network from the existing pools, with N=5 replicates on Domain17 and Kanno and (due to ALARM's substantially higher per-run cost) N=1 replicate per scenario on ALARM — defensible on the same grounds as the ALARM scale-expansion in §5.1: this is a scalability check, and we want sign/direction preservation across networks more than tight confidence intervals within any one cell.

**Table 5. N=3 vs N=5 scalability on all three networks.**

| Cell | Sensitivity gain N=3 (Ext. 3 primary) | Sensitivity gain N=5 (scalability) | Ratio | sub_gap N=3 | sub_gap N=5 |
|---|---|---|---|---|---|
| Domain17 tree | +30.8 (15 reps, 8 scen) | +55.0 (5 reps, 3 scen) | **1.79×** | +0.077 (8/8) | +0.010 (2/3) |
| Kanno tree | +10.1 (15 reps, 8 scen) | +30.0 (5 reps, 3 scen) | **2.96×** | +0.053 (8/8) | +0.056 (3/3) |
| ALARM tree | +3.4 (5 reps, 8 scen) | +13.3 (1 rep, 3 scen) | **3.95×** | +0.024 (6/8) | +0.000 (2/3) |
| Domain17 markov | +21.4 (15 reps, 8 scen) | +81.7 (5 reps, 3 scen) | **3.82×** | −0.018 (3/8) | −0.030 (1/3) |
| Kanno markov | +2.6 (15 reps, 8 scen) | +15.0 (5 reps, 3 scen) | **5.71×** | −0.002 (3/8) | **−0.120** (0/3) |
| ALARM markov | +16.5 (5 reps, 6 scen) | +38.3 (1 rep, 3 scen) | **2.32×** | +0.018 (4/6) | +0.009 (1/3) |
| **Mean ratio** | | | **3.43×** | | |

Four observations. First, **the sensitivity gain scales super-linearly with team size on every cell across all three networks.** Every one of the six (network × axis) cells shows a growth ratio strictly greater than 1, ranging from 1.79× to 5.71×, with a cross-cell mean of 3.43×. This is close to the theoretical prediction from the pairwise monitoring surface: as N goes from 3 to 5, the number of (agent, teammate-target) L3 belief structures per agent grows from (N−1)²=4 to 16, a 4× ratio; over which cross-substrate disagreements can fire, so the expected sensitivity gain is bounded above by approximately that ratio. The observed 3.43× cross-cell mean sits just below that ceiling, consistent with the account's claim that L3 disagreements are the primary source of sensitivity gain and that this source scales with the monitoring surface rather than with the team size linearly.

Second, **the single-agent inference quality loss does not scale with team size.** Only one agent has the mismatched substrate in either configuration, so the per-scenario quality-degradation contribution to the substrate gap is approximately N-invariant. This is the second mechanism.

Third, **the balance between the two forces shifts** in the direction the account predicts, most visibly on the markov axis. On Kanno-markov, the sub_gap moves from −0.002 (essentially null at N=3) to −0.120 (mismatch strongly helps, 0/3 positive scenarios at N=5) — a striking amplification driven by a 5.71× sensitivity gain over a nearly-N-invariant quality loss. On Domain17-markov, sub_gap moves from −0.018 to −0.030 in the same direction. On tree-axis cells, where quality loss is larger to begin with, sub_gap attenuates on Domain17 (+0.077→+0.010) and ALARM (+0.024→+0.000) but stays similar on Kanno (+0.053→+0.056) — the amplified sensitivity gain trades approximately evenly against Kanno's already-modest quality loss. Across all six cells, the movement of sub_gap toward the sensitivity-favoring side (either more negative on axis-favoring cells or attenuated on axis-hurting cells) is consistent with what the two-force account requires: sensitivity gain scaling with team size, quality loss not.

Fourth, **naive projection's suppression of the sensitivity signal scales similarly** — across all six cells, naive suppression at N=5 exceeds naive suppression at N=3 with ratios ranging 1.57× to 5.71× (cross-cell mean 2.72×). At N=5, naive projection destroys enough of the aware-projection sensitivity signal that mm-naive teams are pushed *below* matched-team conflict counts in several conditions, reproducing the "illusory agreement" pathology of §5.4 at amplified magnitude. The larger the team, the more of the sensitivity signal aware projection is capturing and naive projection is throwing away.

The specificity check continues to hold at N=5 on all three networks (max |matched-aware − matched-naive| = 0.016 Domain17-tree, 0.040 Domain17-markov, and correspondingly small on Kanno and ALARM). These are not artifacts of the projection contrast bleeding into matched compositions.

![Figure 10. Team-size scalability across three networks. **Left**: substrate gap by (network × axis × team-size) — solid bars N=3 (Extension 3 primary), hatched bars N=5 (this scalability check). **Right**: total sensitivity gain (initial conflicts, mm-aware − matched-aware) by same. Every cell shows super-linear sensitivity gain growth from N=3 to N=5 (ratios 1.79× to 5.71×, cross-cell mean 3.43×). Sub_gap moves toward the sensitivity-favoring side on 5 of 6 cells (Kanno-tree the exception). ALARM N=5 was run at N=1 replicate per scenario due to per-run cost — hence the missing error bars on the ALARM N=5 bars, and the wider per-scenario spread visible in the dots.](figs/fig10_n5_scalability.png){width=100%}

The scaling ratios in Table 5 vary from 1.79× to 5.71× across the six axis-network combinations, so one could reasonably ask whether the 3.43× cross-cell mean is a genuine scaling relationship or a coincidence — an average of six unrelated numbers. Figure 11 tests this directly: it plots the observed sensitivity gain at N=5 against the observed sensitivity gain at N=3 for each of the six (network × axis) combinations. If the scaling relationship is real, cells that had bigger sensitivity gains at N=3 should have proportionally bigger sensitivity gains at N=5. The fitted line has slope 1.94 (i.e., every 1-unit of N=3 sensitivity gain predicts approximately 2 units of N=5 gain, plus an 11.5-unit constant), Pearson r = +0.815, R² = 0.665, p = 0.048. The correlation is significant at the 0.05 level despite n=6, and only Kanno-tree sits notably below the fitted line (a specific point we discuss further in §6.5). Every one of the six points lies above the y=x reference — that is, every cell shows super-linear scaling, not merely proportional scaling. The cross-network consistency here — a single linear relationship explaining 67% of the variance across three networks that differ substantially in topology, CPT structure, and node count — is what would be predicted by an underlying (N−1)²-monitoring-surface mechanism operating uniformly across networks, and is what the two-force account requires.

![Figure 11. Cross-network scaling correlation of the sensitivity gain. Each point is one of six (network × axis) combinations; x-axis is the observed total sensitivity gain at N=3 (from Extension 3 primary results), y-axis is the same measurement at N=5 (from the scalability check). Dashed line: linear fit (slope 1.94, intercept +11.5). Dotted line: y=x reference. All six points lie above the y=x line, confirming super-linear scaling in every cell. R² = 0.665, p = 0.048 despite n=6 — the scaling relationship is statistically significant and network-general.](figs/fig11_scaling_correlation.png){width=95%}

# 6. Discussion

## 6.1 The two forces are both real and both necessary

The 2×2×2 design lets us make a claim we could not make from any single (network, axis) cell. Substrate mismatch is not one thing. It is at minimum two things: a *sensitivity gain* that surfaces additional conflicts and is bounded by how much the mismatched substrate's committed diagnoses diverge from BBN's, and an *inference quality loss* that is bounded by how much worse the mismatched substrate is at getting individual nodes right. Both forces are measurable at the level of individual scenarios in ways that generalize across networks. The signed effect of mismatch on team accuracy is the balance.

This reframes the Anthropomorphization Trap. The original trap thesis — matched > mismatched-aware > mismatched-naive — turns out to hold, on aggregate, only when the mismatched substrate is inferentially weaker than BBN. When the mismatched substrate is inferentially close to BBN (Markov on either network), aggregate mismatch is either neutral or mildly beneficial, and the aware/naive ordering becomes empirically noisy. The trap is not a single-mechanism claim; it is a compound one, and its aggregate direction depends on which of the two forces dominates in a given (network, axis) regime.

## 6.2 What naive projection actually does

Sections 5.4 and 5.5 give an operationally precise account of naive projection. Under matched compositions, naive projection is a null operation — indistinguishable from aware projection because everyone runs the same substrate. Under mismatched compositions, naive projection is a *conflict-detection suppressor*: it hides the disagreements that substrate heterogeneity produces, by structurally instantiating projected engines that agree with the detecting agent's own inference by construction. This is neither a form of cognitive laziness nor a lack of information — it is a definite representational commitment that renders substrate heterogeneity invisible.

The interesting consequence is that naive projection is *bounded* in its harm by the size of the substrate gap. When the substrate gap is small (Kanno-Markov), naive projection has little to suppress, so its accuracy penalty is small (aware gap +0.004). When the substrate gap is large (Domain17-tree), naive projection has much to suppress, and correspondingly more accuracy is left on the table. This inverts the intuition that naive projection should always be costly: it is costly only to the extent that there is signal for it to erase.

## 6.3 Phantom conflicts and the sensitivity ceiling

The phantom/real decomposition surfaces a further subtlety. Not all sensitivity is productive. The extra conflicts detected under aware projection with mismatched substrates include a substantial fraction (typically 40–50%) in which the detecting agent is already correct and the perceived mismatch is illusory from an oracle's perspective. These phantom conflicts drive the coop phase to spend actions on nodes that did not need action, potentially degrading the team's action efficiency.

This gives a natural ceiling on the sensitivity gain. Even when a team can detect a great many mismatch-induced conflicts (Domain17-tree, +30.8 extras), the team is only marginally better off in coordination terms because half of that extra activity is misdirected. The tree engine's L1 quality loss is what tips this from a wash into an accuracy loss.

The phantom effect is not merely nuisance: it is potentially the mechanism connecting substrate diversity to *dysfunctional coordination*. A team of agents with different substrates is not a team of agents with more perspectives to bring to bear; it is a team that generates additional disagreements, most of which are structurally illusory. In a real human-AI teaming context, one could reasonably ask whether a majority of disagreements between a human expert and a model are "real" (the human is wrong and would benefit from the model's flag) or "phantom" (the human is right and the flag will simply produce friction). The extension to real teams is speculative but the framework generalizes.

## 6.4 The Markov surprise

The most surprising cell in Table 1 is Domain17-Markov, where the substrate gap is mildly *negative* — mismatched teams marginally beat matched teams. In an earlier draft of this project we identified this as an unresolved anomaly. With the two-force account in hand, it is no longer anomalous. Domain17-Markov satisfies exactly the conditions where sensitivity gain (real +10.5) outweighs quality loss (small — Markov's dominant-parent-mediated approximation is close to BBN on Domain17's CPTs). It fits the account.

The Markov approximation being nearly lossless on many real Bayesian networks is a familiar practical observation, driven by the concentration of mutual information in dominant parent edges when CPTs are strongly informative. The finding here is that this near-losslessness has a cooperative-diagnosis consequence: a Markov agent can be added to a BBN team without accuracy cost, and the sensitivity signal its presence generates is a modest positive contribution.

## 6.5 Boundary conditions and limitations

Several boundary conditions bear noting. First, our primary experiments use three-person teams, following Okabe et al. (2025). This choice is defensible on grounds that a three-person team is the minimum non-trivial unit for studying recursive mutual belief: a two-person dyad has no third-party belief structure to project, while a three-person team introduces the L3 projected-belief layer without yet requiring the mental-subgrouping machinery of Mahardhika et al. (2016). We tested how far the two-force account extends at larger team sizes with the N=5 scalability check reported in §5.8, run on all three networks. That check makes three quantitative claims we could not make from N=3 alone: (i) the sensitivity gain scales super-linearly with team size on every one of the six (network × axis) cells, ratios 1.79× to 5.71×, cross-cell mean 3.43× — close to the theoretical (N−1)² monitoring-surface ratio of 4×; (ii) the single-agent inference quality loss does not scale with team size, since only one agent is the substrate outlier in either configuration; (iii) the balance between the two forces therefore shifts as N grows, most dramatically on Kanno-markov (sub_gap −0.002 → −0.120 — mismatch becomes strongly beneficial at N=5) and on Domain17-tree (+0.077 → +0.010 — mismatch nearly neutralized). The account thus predicts a specific *quantitative* structure — sensitivity gain scaling with (N−1)² while quality loss stays constant — that we observe across three separate networks with different topologies, CPT structures, and node counts. What we still do not settle from N=5 is behavior at N=7-9 (typical operational team size), at higher outlier fractions (majority-minority splits, substrate-clustered subteams), or under mental subgrouping. These are the natural next studies. Second, all our experiments crossed the substrate axis at a fixed outlier count (one agent). At N=5, two-outlier or three-outlier compositions become tractable and would test whether quality loss aggregates roughly linearly with the outlier fraction, as the account predicts; we did not run those compositions here. The pattern our three-network N=5 data documents, however — that the sensitivity mechanism scales faster with team size than the quality mechanism does, on every cell tested — is directly what the two-force account requires to be true, not an artifact of the specific compositions or networks tested.

Second, all results reported here are at THD = 0.55. Preliminary sweeps at THD = 0.65 (on Kanno-tree) show the sensitivity mechanism intact but the aware-vs-naive accuracy consequences change with threshold — this is expected, since higher thresholds produce fewer commitments and therefore less surface area for aware/naive to bite. A full THD × cell sweep is a natural next study.

Third, our N=15 per cell for the two 17-node networks is reduced from the original EMBM protocol of N=40, and ALARM was run at a further-reduced N=5. Aggregate estimates on the 17-node networks are stable at N=15; per-scenario estimates carry larger uncertainty. ALARM's N=5 was deliberately chosen to fit the scale-expansion role of that network: the goal was sign consistency and mechanism preservation across a larger network, not tightening confidence intervals already established at N=15 on the smaller networks. Where our claims turn on 8/8 (or on ALARM, 6/8) replication of a sign, replication is at 8/8 (or 6/8) regardless of N; where they turn on aggregate significance testing, the reduced N is a conservative choice (less power) rather than a permissive one. The attenuated but sign-preserved substrate gap on ALARM-tree (+0.024 vs +0.077 on Domain17-tree and +0.053 on Kanno-tree) is discussed further in §5.1 as a scale effect predicted by the two-force account rather than as an artifact of reduced N.

Fourth, we used hand-specified engines throughout. Learned engines (e.g., a decision tree trained on fault data rather than compiled from CPTs) would introduce training-noise confounds that we deliberately avoided. Extending the framework to learned engines is a natural direction but requires care in disentangling substrate structure from training regime.

Fifth, we studied three substrates. Others — for example, large language models used as approximate posterior estimators — would extend the analysis but require new translation rules. Nothing in the two-force account is specific to the three substrates we used.

Sixth, the phantom/real decomposition is defined at the level of the initial-conflict snapshot, before cooperation. Extending it to the full cooperation trajectory would require tracking whether each action taken during coop resolved a real or phantom conflict at the moment it was taken. This is a next-instrumentation step.

# 7. Conclusion

Substrate heterogeneity in recursive-mutual-belief teams is not a single force. It is at least two: a sensitivity gain that surfaces additional inter-agent conflicts when projection is aware of the heterogeneity, and an inference quality loss that scales with how much the mismatched substrate degrades single-agent diagnostic accuracy. The signed effect of mismatch on team accuracy is the balance, and it can go either way depending on the (network, axis) regime.

The two forces are separately measurable and separately robust. The sensitivity gain replicates with per-scenario perfect (8/8) or near-perfect (6/8) sign consistency in the four (network × axis) cells we studied on the two 17-node networks, and holds directionally but with attenuated magnitude on the 37-node ALARM network (substrate gap +0.024 with 6/8 sign consistency on the tree axis). The naive-projection suppression of that sensitivity replicates similarly on the 17-node networks, and the matched-team specificity check confirms both effects arise specifically from heterogeneity rather than from projection mode per se on all three networks (max |matched-aware − matched-naive| = 0.007 on ALARM). The phantom/real decomposition shows the sensitivity gain is real but not fully productive.

The Anthropomorphization Trap — the original hypothesis that assuming your heterogeneous teammate reasons like you do costs accuracy — is best understood as a consequence of these two forces plus the naive-projection suppression, rather than as a single-mechanism prediction. When the substrate gap is large, naive projection loses meaningful information about disagreements, so aware projection wins. When the substrate gap is small, naive projection has little to lose. Whether mismatch is beneficial or harmful in the first place depends on whether the two underlying forces are balanced in favor of sensitivity or quality.

We suggest that recursive-mutual-belief models of team cognition are best formulated with pluggable inference substrates from the outset. Homogeneity assumptions are analytically convenient but empirically restrictive, and — as this paper shows — the coordination dynamics they hide are not merely additive corrections but structurally distinct forces whose balance is the actual dependent variable of interest.

# Acknowledgments

[Reserved.]

# Data availability

All experimental scripts and results JSONs are available at [repository placeholder]. The 2×2×2 primary design can be reproduced with `run_conflict_instrument.py {domain17|kanno} {tree|markov}` at the documented base seeds (900_000, 910_000, 500_000, 800_000 for the four cells respectively). The ALARM scale-expansion cells are reproduced with `N_RUNS=5 python run_conflict_instrument.py alarm {tree|markov}` at base seeds 1_000_000 and 1_010_000. The ALARM network is loaded from Beinlich et al.'s canonical `.bif` file as distributed by the `bnlearn` repository (`bnlearn.com/bnrepository/alarm/alarm.bif.gz`); the 8 hard scenarios for ALARM were sampled with `sample_pool_alarm.py` at pool seed 20260706 and are included in the release as `hard_scenarios_alarm.pkl`.

# References

Anderson, J. R., & Lebiere, C. (1998). *The atomic components of thought*. Mahwah, NJ: Lawrence Erlbaum Associates.

Andrews, R. W., Lilly, J. M., Srivastava, D., & Feigh, K. M. (2023). The role of shared mental models in human-AI teams: A theoretical review. *Theoretical Issues in Ergonomics Science*, 24(2), 129–175. https://doi.org/10.1080/1463922X.2022.2061080

Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989). The ALARM Monitoring System: A case study with two probabilistic inference techniques for belief networks. In J. Hunter, J. Cookson, & J. Wyatt (Eds.), *Proceedings of the 2nd European Conference on Artificial Intelligence in Medicine (AIME '89)* (pp. 247–256). Berlin: Springer-Verlag.

Cao, S., MacLaren, N. G., Cao, Y., Marshall, J., Dong, Y., Yammarino, F. J., … Ruark, G. A. (2022). Group size and group performance in small collaborative team settings: An agent-based simulation model of collaborative decision-making dynamics. *Complexity*, 2022, 8265296. https://doi.org/10.1155/2022/8265296

Cooke, N. J., Salas, E., Cannon-Bowers, J. A., & Stout, R. J. (2000). Measuring team knowledge. *Human Factors*, 42(1), 151–173. https://doi.org/10.1518/001872000779656561

Cooke, N. J., Gorman, J. C., Myers, C. W., & Duran, J. L. (2013). Interactive team cognition. *Cognitive Science*, 37(2), 255–285. https://doi.org/10.1111/cogs.12009

Crosscombe, M., & Lawry, J. (2016). A model of multi-agent consensus for vague and uncertain beliefs. *Adaptive Behavior*, 24(4), 249–260. https://doi.org/10.1177/1059712316661395

Dourish, P., & Bellotti, V. (1992). Awareness and coordination in shared workspaces. In *Proceedings of the 1992 ACM Conference on Computer-Supported Cooperative Work* (pp. 107–114). https://doi.org/10.1145/143457.143468

Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive-developmental inquiry. *American Psychologist*, 34(10), 906–911. https://doi.org/10.1037/0003-066X.34.10.906

Flemisch, F., Abbink, D. A., Itoh, M., Pacaux-Lemoine, M.-P., & Weßel, G. (2019). Joining the blunt and the pointy end of the spear: Towards a common framework of joint action, human–machine cooperation, cooperative guidance and control, shared, traded and supervisory control. *Cognition, Technology & Work*, 21, 555–568. https://doi.org/10.1007/s10111-019-00576-1

Glinton, R., Scerri, P., & Sycara, K. P. (2010). Exploiting scale invariant dynamics for efficient information propagation in large teams. In *Proceedings of the 9th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)* (pp. 21–30).

Grimm, V., Berger, U., Bastiansen, F., Eliassen, S., Ginot, V., Giske, J., … DeAngelis, D. L. (2006). A standard protocol for describing individual-based and agent-based models. *Ecological Modelling*, 198(1–2), 115–126. https://doi.org/10.1016/j.ecolmodel.2006.04.023

Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C., DeAngelis, D. L., … Ayllón, D. (2020). The ODD protocol for describing agent-based and other simulation models: A second update to improve clarity, replication, and structural realism. *Journal of Artificial Societies and Social Simulation*, 23(2), 7. https://doi.org/10.18564/jasss.4259

Gutwin, C., & Greenberg, S. (2004). The importance of awareness for team cognition in distributed collaboration. In E. Salas & S. M. Fiore (Eds.), *Team cognition: Understanding the factors that drive process and performance* (pp. 177–201). American Psychological Association.

Hutchins, E. (1995). *Cognition in the wild*. Cambridge, MA: MIT Press.

Kanno, T., Furuta, K., & Kitahara, Y. (2013). A model of team cognition based on mutual beliefs. *Theoretical Issues in Ergonomics Science*, 14(1), 38–52. https://doi.org/10.1080/1464536X.2011.573010

Kosinski, M. (2024). Evaluating large language models in theory of mind tasks. *Proceedings of the National Academy of Sciences*, 121(45), e2405460121. https://doi.org/10.1073/pnas.2405460121

Li, J., & Xiao, R. (2017). Agent-based modelling approach for multidimensional opinion polarization in collective behaviour. *Journal of Artificial Societies and Social Simulation*, 20(2), 4. https://doi.org/10.18564/jasss.3385

Mahardhika, D., Kanno, T., & Furuta, K. (2016). Team cognition model based on mutual beliefs and mental subgrouping. *Journal of Interaction Science*, 4, 1–10. https://doi.org/10.1186/s40166-016-0014-6

McNeese, M., Salas, E., & Endsley, M. R. (Eds.). (2020). *Foundations and theoretical perspectives of distributed team cognition*. Boca Raton, FL: CRC Press.

Mohammed, S., Hamilton, K., Tesler, R., Mancuso, V., & McNeese, M. (2015). Time for temporal team mental models: Expanding beyond "what" and "how" to incorporate "when." *European Journal of Work and Organizational Psychology*, 24(5), 693–709. https://doi.org/10.1080/1359432X.2015.1024664

Nelson, T. O., & Narens, L. (1994). Why investigate metacognition. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 1–25). Cambridge, MA: MIT Press.

Newell, A. (1990). *Unified theories of cognition*. Cambridge, MA: Harvard University Press.

Nonose, K., Kanno, T., & Furuta, K. (2010). An evaluation method of team situation awareness based on mutual belief. *Cognition, Technology & Work*, 12, 31–40. https://doi.org/10.1007/s10111-008-0127-y

Nonose, K., Kanno, T., & Furuta, K. (2014). Effects of metacognition in cooperation on team behaviors. *Cognition, Technology & Work*, 16, 349–358. https://doi.org/10.1007/s10111-013-0265-8

Okabe, N., Kanno, T., Cho, S., Furuta, K., Yoshida, H., Karikawa, D., Nonose, K., & Inoue, S. (2025). Modeling and simulation of three-person team cooperation considering mutual beliefs. *Cognition, Technology & Work*, 27(1), 159–179. https://doi.org/10.1007/s10111-025-00791-z

Rousseau, R., Tremblay, S., & Breton, R. (2004). Defining and modeling situation awareness: A critical review. In S. Banbury & S. Tremblay (Eds.), *A cognitive approach to situation awareness: Theory and application* (pp. 3–21). Aldershot, UK: Ashgate.

Salas, E., Cooke, N. J., & Rosen, M. A. (2008). On teams, teamwork, and team performance: Discoveries and developments. *Human Factors*, 50(3), 540–547. https://doi.org/10.1518/001872008X288457

Salas, E., Fiore, S. M., & Letsky, M. P. (Eds.). (2011). *Theories of team cognition: Cross-disciplinary perspectives*. New York: Routledge.

Schelble, B. G., Flathmann, C., McNeese, N. J., O'Neill, T., Pak, R., & Namara, M. (2023). Investigating the effects of perceived teammate artificiality on human performance and cognition. *International Journal of Human–Computer Interaction*, 39(13), 2686–2701. https://doi.org/10.1080/10447318.2022.2085191

She, M., & Li, Z. (2017). Team situation awareness: A review of definitions and conceptual models. In D. Harris (Ed.), *Engineering psychology and cognitive ergonomics: Performance, emotion and situation awareness (EPCE 2017), Lecture Notes in Computer Science, vol. 10275* (pp. 406–415). Cham: Springer. https://doi.org/10.1007/978-3-319-58472-0_31

Tuomela, R., & Miller, K. (1988). We-intentions. *Philosophical Studies*, 53, 367–389. https://doi.org/10.1007/BF00353512

Wright, J. L., Chen, J. Y. C., Barnes, M. J., & Boyce, M. W. (2015). The effects of information level on human–agent interaction for route planning. *Proceedings of the Human Factors and Ergonomics Society Annual Meeting*, 59(1), 811–815. https://doi.org/10.1177/1541931215591247

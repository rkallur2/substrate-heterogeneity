# Substrate Heterogeneity in EMBM Cooperative Diagnosis

Replication package for:

> Kalluri, R. & Kanno, T. (2026). **Two Forces of Substrate Heterogeneity: Sensitivity Gain and Quality Loss in Multi-Agent Cooperative Diagnosis.** *Manuscript in preparation.*

This repository extends the Extended Mutual Belief Model (EMBM) of Okabe et al. (2025) with pluggable inference substrates and runs a 2×2×2 experimental design (network × mismatched-substrate axis × projection mode) on two 17-node Bayesian networks. It provides all simulation code, the raw results JSONs, and the figures used in the manuscript.

**Primary reference for the underlying model:** Okabe, N., Kanno, T., Cho, S., Furuta, K., Yoshida, H., Karikawa, D., Nonose, K., & Inoue, S. (2025). Modeling and simulation of three-person team cooperation considering mutual beliefs. *Cognition, Technology & Work*, 27(1), 159–179. [DOI: 10.1007/s10111-025-00791-z](https://doi.org/10.1007/s10111-025-00791-z)

---

## Repository layout

```
substrate_abm_release/
├── README.md                    (this file)
├── requirements.txt             (Python dependencies)
├── LICENSE                      (fill in your choice)
├── .gitignore
│
├── *.py                         (all model code + experiment runners at root)
├── agent0.dne, agent1.dne, agent2.dne   (Netica car-diagnosis networks)
│
├── results/                     (scenario pools + per-experiment JSONs)
│   ├── hard_scenarios.pkl                 pool of 8 hard scenarios for Kanno
│   ├── hard_scenarios_domain17.pkl        pool of 8 hard scenarios for Domain17
│   ├── conflict_instrument_*.json         2×2 with conflict-count instrumentation
│   ├── dual_metric_domain17_markov.json   full vs. abstention-aware metric
│   ├── aggregate_2x2x2.json               consolidated aggregate across cells
│   └── *.json                             (other single-condition JSONs)
│
├── figures/                     (all manuscript PNGs)
│   ├── fig1_substrates.png
│   ├── fig2_projection.png
│   ├── fig3_conflict_types.png
│   ├── fig4_networks.png                  Kanno & Domain17 topology diagrams
│   ├── fig5_substrate_gap.png             main outcome
│   ├── fig6_conflict_counts.png           mechanism: sensitivity gain
│   ├── fig7_phantom_real.png              phantom/real decomposition
│   └── fig8_two_force.png                 two-force account visualized
│
└── paper/
    ├── paper_substrate_heterogeneity.md   markdown source
    └── paper_substrate_heterogeneity.docx rendered manuscript
```

## Requirements

Python 3.9+. Install dependencies:

```bash
pip install -r requirements.txt
```

The model itself only needs `numpy`; `scipy` is used for the significance tests in the analysis snippets.

## Quick start

Run one 2×2 with conflict-count instrumentation, from the repository root:

```bash
python run_conflict_instrument.py domain17 markov
```

Results append incrementally to `results/conflict_instrument_domain17_markov.json`. The script checkpoints per scenario so it is safe to interrupt and restart. Approximate wall time per (network, axis) at N=15 runs per cell: 4–6 minutes.

Full 2×2×2 sweep:

```bash
python run_conflict_instrument.py kanno tree       # ~3 min
python run_conflict_instrument.py kanno markov     # ~5 min
python run_conflict_instrument.py domain17 tree    # ~5 min
python run_conflict_instrument.py domain17 markov  # ~5 min
```

## Reproducing the paper's numbers

Every result JSON in `results/` is regenerated deterministically from the seed convention documented at the top of each runner script. Base seeds used in the manuscript:

| Cell | Runner | Base seed |
|---|---|---|
| Kanno · tree | `run_multiscen_2x2.py`, `run_conflict_instrument.py kanno tree` | 500,000 |
| Kanno · markov | `run_markov_2x2.py`, `run_conflict_instrument.py kanno markov` | 800,000 |
| Domain17 · tree | `run_domain17_2x2.py tree`, `run_conflict_instrument.py domain17 tree` | 900,000 |
| Domain17 · markov | `run_domain17_2x2.py markov`, `run_conflict_instrument.py domain17 markov` | 910,000 |

Runs at N=15 per cell (the manuscript's default) will exactly match the JSONs in `results/` at these seeds.

## The two hard-scenario pools

Both pools are pre-sampled and stored as pickles:

- `results/hard_scenarios.pkl` — 8 Kanno scenarios (hardness 7–9), sampled from a pool of 120 fault configurations at `pool_seed=42`. Distribution splits 4-way on the BatVolt fault state (4 `dead`, 4 `weak`).
- `results/hard_scenarios_domain17.pkl` — 8 Domain17 scenarios (hardness 12–15), sampled from a pool of 120 at the same seed.

To regenerate the pools from scratch (if you change the sampling protocol):

```bash
python sample_pool.py            # regenerates hard_scenarios.pkl
python sample_pool_domain17.py   # regenerates hard_scenarios_domain17.pkl
```

## The core model

- **`bbn_engine.py`** — Bayesian belief network with exact belief propagation. Provides `CausalSpec`, `BBNEngine`, and the `InferenceEngine` interface (`ingest`, `posterior`, `diagnosis`).
- **`markov_engine.py`** — Hidden-Markov-style engine derived by linearizing the DAG into a causal ordering.
- **`tree_engine.py`** — Decision-tree engine compiled from CPTs with leaf pseudo-posteriors.
- **`embm_agent.py`** — Three-layer EMBM agent (self L1, direct L2, projected L3), with `find_conflicts` implementing the Type I/II/III conflict detection.
- **`experiment.py`** — Team building, cooperation-phase loop, per-run outcome recording.
- **`simulation.py`** — Scenario sampling, node-importance calculation (mutual information sum, per Okabe et al. Eq. 3), team-accuracy metric.
- **`translation.py`** — Cross-substrate evidence translation (BBN posterior → tree argmax, tree label → soft BBN evidence, etc.).
- **`domain17.py`** — Hand-specified 17-node synthetic causal spec (`make_domain17()`, `make_observability17()`).
- **`domain_real.py`** — Loader for the Kanno network from `agent1.dne` via Netica import.
- **`netica_import.py`** — Netica `.dne` file parser.

## Notes on Kanno network files

`agent0.dne`, `agent1.dne`, `agent2.dne` are three variations of the Kanno car-diagnosis network with the same 17-node structure but differing conditional probability tables. The paper uses the shared-baseline configuration (`agent1.dne` for all three agents) throughout. `agent0/2.dne` are retained for the parametric-heterogeneity extension described in `domain_real.py` but not used in the current manuscript.

## Citation

If you use this repository, please cite both the primary EMBM reference and this work:

```bibtex
@article{okabe2025modeling,
  title={Modeling and simulation of three-person team cooperation considering mutual beliefs},
  author={Okabe, Naoki and Kanno, Taro and Cho, Sumie and Furuta, Kazuo and Yoshida, Haruka and Karikawa, Daisuke and Nonose, Kohei and Inoue, Satoru},
  journal={Cognition, Technology \& Work},
  volume={27},
  number={1},
  pages={159--179},
  year={2025},
  publisher={Springer},
  doi={10.1007/s10111-025-00791-z}
}

@unpublished{kalluri2026two,
  title={Two Forces of Substrate Heterogeneity: Sensitivity Gain and Quality Loss in Multi-Agent Cooperative Diagnosis},
  author={Kalluri, Ravikiran and Kanno, Taro},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

See `LICENSE`. The Netica `.dne` files are derived from the original Norsys Software Corp. car-diagnosis example and are included with the understanding that they are educational-example files; commercial redistribution may require permission from Norsys.

## Contact

For questions about this replication package, contact Ravikiran Kalluri (Northeastern University). For questions about the underlying EMBM framework, contact Taro Kanno (University of Tokyo).

# Two Forces of Substrate Heterogeneity: Manuscript Package

**Paper**: *Two Forces of Substrate Heterogeneity: Sensitivity Gain and Quality Loss in Multi-Agent Cooperative Diagnosis*
**Authors**: Ravikiran Kalluri (Northeastern), Taro Kanno (U Tokyo)
**Target venue**: JASSS

This package contains the full manuscript, all figures, all reproducible code, all raw and aggregate experimental results, and canonical network source files.

## Contents

```
substrate_manuscript_full/
├── README.md                                    (this file)
├── LICENSE                                      (MIT)
├── requirements.txt                             (Python dependencies)
├── paper/
│   ├── paper_substrate_heterogeneity.md         (source markdown, ~14.5k words)
│   ├── paper_substrate_heterogeneity.docx       (rendered Word document, 37 pages)
│   ├── paper_substrate_heterogeneity.pdf        (rendered PDF)
│   └── figures/                                 (11 PNG figures embedded in the paper)
│       ├── fig1_substrates.png                  (three inference substrates schematic)
│       ├── fig2_projection.png                  (aware vs naive projection matrix)
│       ├── fig3_conflict_types.png              (Type I/II/III metacognitive conflicts)
│       ├── fig4_networks.png                    (Kanno and Domain17 topologies)
│       ├── fig5_alarm.png                       (ALARM 37-node topology, Ext. 3)
│       ├── fig6_substrate_gap.png               (Fig 6 in text: substrate gap bars)
│       ├── fig7_conflict_counts.png             (Fig 7: init conflict counts)
│       ├── fig8_phantom_real.png                (Fig 8: phantom vs real decomposition)
│       ├── fig9_two_force.png                   (Fig 9: two-force scatter across 6 cells)
│       ├── fig10_n5_scalability.png             (Fig 10: N=3 vs N=5 across 3 networks)
│       └── fig11_scaling_correlation.png        (Fig 11: cross-network scaling R²=0.665)
├── code/                                        (Python 3.x — 36 files)
│   ├── — Inference substrates and translation —
│   ├── bbn_engine.py                            (exact Bayesian propagation, junction-tree)
│   ├── markov_engine.py                         (hidden-Markov causal-linearization approx)
│   ├── tree_engine.py                           (decision-tree argmax-on-features approx)
│   ├── translation.py                           (cross-substrate belief translation)
│   ├── — Agent architecture and simulation core —
│   ├── embm_agent.py                            (EMBM agent with 7 (N=3) or 21 (N=5) engines)
│   ├── experiment.py                            (build_team, run_once, TeamConfig)
│   ├── simulation.py                            (scenarios, node importance, sample_scenario)
│   ├── — Network specifications —
│   ├── domain.py                                (base CausalSpec)
│   ├── domain17.py                              (17-node synthetic; N=3 and N=5 partitions)
│   ├── domain_real.py                           (Kanno 17-node; loads agent1.dne)
│   ├── alarm_domain.py                          (ALARM 37-node; loads alarm.bif)
│   ├── — Canonical network files —
│   ├── alarm.bif                                (Beinlich et al. 1989, from bnlearn)
│   ├── agent0.dne, agent1.dne, agent2.dne       (Kanno Netica specs from Okabe et al.)
│   ├── — Scenario samplers —
│   ├── sample_pool.py                           (Kanno pool)
│   ├── sample_pool_domain17.py                  (Domain17 pool)
│   ├── sample_pool_alarm.py                     (ALARM pool with clinical hardness)
│   ├── — Primary experiment runners —
│   ├── run_conflict_instrument.py               (Ext. 3 primary 2x2x2 across 3 networks)
│   ├── run_n5_instrument.py                     (Ext. 2 N=5 scalability across 3 networks)
│   ├── — Analysis and phantom/real instrumentation —
│   ├── count_conflicts.py
│   ├── instrument_tree_correctness.py
│   ├── diag_null_result.py
│   ├── netica_import.py, variations.py
│   ├── — Legacy runners (early 2x2 designs, kept for full reproducibility) —
│   ├── run_domain17_2x2.py, run_dual_metric.py, run_markov_2x2.py
│   ├── run_multiscen_2x2.py, run_multiscen_2x2_thd065.py, run_phase1.py
│   ├── run_real_2x2.py, run_real_2x2_hard.py, run_real_multiscen.py
│   └── — Figure generators —
│       ├── make_fig10_n5.py                     (Fig 10: N=3 vs N=5 bars)
│       └── make_fig11_correlation.py            (Fig 11: cross-network scaling R²)
├── results/                                     (raw JSONs + aggregate pickles)
│   ├── — Primary Extension 3 results (N=3 team size) —
│   ├── conflict_instrument_domain17_tree.json   (8 scen × N=15 reps)
│   ├── conflict_instrument_domain17_markov.json
│   ├── conflict_instrument_kanno_tree.json      (8 scen × N=15 reps)
│   ├── conflict_instrument_kanno_markov.json
│   ├── conflict_instrument_alarm_tree.json      (8 scen × N=5 reps, scale-expansion)
│   ├── conflict_instrument_alarm_markov.json
│   ├── — Extension 2 N=5 scalability results —
│   ├── n5_instrument_domain17_tree.json         (3 scen × N=5 reps)
│   ├── n5_instrument_domain17_markov.json
│   ├── n5_instrument_kanno_tree.json            (3 scen × N=5 reps)
│   ├── n5_instrument_kanno_markov.json
│   ├── n5_instrument_alarm_tree.json            (3 scen × N=1 rep, cost-limited)
│   ├── n5_instrument_alarm_markov.json
│   ├── n5_three_network_aggregate.pkl           (Fig 11 source data)
│   ├── — Hard scenario pools —
│   ├── hard_scenarios.pkl                       (Kanno, 8 hard scenarios, pool_seed=42)
│   ├── hard_scenarios_domain17.pkl              (Domain17, 8 hard, pool_seed=20260624)
│   ├── hard_scenarios_alarm.pkl                 (ALARM, 8 hard, pool_seed=20260706)
│   ├── — Supplementary —
│   ├── aggregate_2x2x2.json                     (early 2x2x2 aggregate)
│   ├── dual_metric_domain17_markov.json         (§5.2 metric-dependence check)
│   └── tree_correctness.json                    (tree substrate validation)
└── docs/
    └── architecture.md                          (design notes on the ABM architecture)
```

## Reproduction: how to regenerate every result in the paper

**Setup**:
```bash
pip install -r requirements.txt
```

**Extension 3 primary results (§5.1–§5.7, N=3 team size)**:
```bash
cd code
python sample_pool_domain17.py     # produces hard_scenarios_domain17.pkl (skip if already there)
python sample_pool.py              # Kanno pool (hard_scenarios.pkl)
python sample_pool_alarm.py        # ALARM pool

# Primary 6 cells (base_seed values documented in each script header)
python run_conflict_instrument.py domain17 tree      # writes conflict_instrument_domain17_tree.json
python run_conflict_instrument.py domain17 markov
python run_conflict_instrument.py kanno    tree
python run_conflict_instrument.py kanno    markov
N_RUNS=5 python run_conflict_instrument.py alarm tree
N_RUNS=5 python run_conflict_instrument.py alarm markov
```

**Extension 2 N=5 scalability results (§5.8)**:
```bash
N_RUNS=5 python run_n5_instrument.py domain17 tree
N_RUNS=5 python run_n5_instrument.py domain17 markov
N_RUNS=5 python run_n5_instrument.py kanno    tree
N_RUNS=5 python run_n5_instrument.py kanno    markov
N_RUNS=1 python run_n5_instrument.py alarm    tree      # cost-limited to N=1 rep
N_RUNS=1 python run_n5_instrument.py alarm    markov
```

**Regenerate result figures (Fig 6-9)**:
Included as separate figure-generation scripts once the JSONs above exist.
`make_fig10_n5.py` and `make_fig11_correlation.py` are self-contained.

Base seeds used (documented in each runner):
- Domain17 tree/markov: 900,000 / 910,000
- Kanno tree/markov: 500,000 / 800,000
- ALARM tree/markov: 1,000,000 / 1,010,000
- N=5 Domain17 tree/markov: 1,100,000 / 1,110,000
- N=5 Kanno tree/markov: 1,200,000 / 1,210,000
- N=5 ALARM tree/markov: 1,300,000 / 1,310,000

## Data at a glance

**Substrate gap by (network × axis) at N=3** (see Table 1, §5.1):

| Network  | Tree axis  | Markov axis |
|----------|------------|-------------|
| Domain17 | +0.077 (8/8) | −0.019 (3/8) |
| Kanno    | +0.053 (8/8) | −0.002 (4/8) |
| ALARM    | +0.024 (6/8) | +0.008 (4/8) |

**Sensitivity gain scaling N=3 → N=5** (see Table 5 and Fig 11, §5.8):

| Cell            | N=3 gain | N=5 gain | Ratio |
|-----------------|----------|----------|-------|
| Domain17 tree   |   +30.8  |  +55.0   | 1.79× |
| Kanno tree      |   +10.1  |  +30.0   | 2.96× |
| ALARM tree      |    +3.4  |  +13.3   | 3.95× |
| Domain17 markov |   +21.4  |  +81.7   | 3.82× |
| Kanno markov    |    +2.6  |  +15.0   | 5.71× |
| ALARM markov    |   +16.5  |  +38.3   | 2.32× |
| **Cross-cell mean** |      |          | **3.43×** |

Cross-network scaling correlation (Fig 11): r = +0.815, R² = 0.665, p = 0.048.

## License

MIT (see LICENSE file). Please cite the paper and Okabe et al. (2025) EMBM if using this code.

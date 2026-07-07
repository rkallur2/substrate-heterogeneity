"""
run_phase1.py
=============
Entry point: reproduces the Phase-1 validation -- the five Okabe/Kanno model
variations on easy and hard scenarios, printing final accuracy, actions to
convergence, and variance.

A faithful reimplementation should show:
  * non-MB0 plateaus low (sees only its own nodes)
  * non-MB1 reaches the information-sharing ceiling
  * MB models converge via mutual-belief reasoning with run-to-run variance
    driven by action ordering (the stability dimension)

Usage:  python3 run_phase1.py
"""

import random
import numpy as np
from domain import make_domain, make_observability
from simulation import SimConfig, sample_scenario, Scenario
from variations import run_variation_many

VARIATIONS = ["non-MB0", "non-MB1", "MB0", "MB1", "MB2"]


def pick_scenario(spec, rng, difficulty):
    """Sample a scenario of a target difficulty.
    'easy'  : system healthy, power ok (priors already favour the truth)
    'hard'  : impaired system with a dead power source (truth far from priors)"""
    for _ in range(500):
        s = sample_scenario(spec, rng, difficulty)
        if difficulty == "easy" and s.truth["SystemHealth"] == "healthy" \
                and s.truth["PowerSource"] == "ok":
            return s
        if difficulty == "hard" and s.truth["SystemHealth"] == "impaired" \
                and s.truth["PowerSource"] == "dead":
            return s
    return s  # fallback


def main(n_runs=100, thd=0.75, seed=2025):
    spec = make_domain()
    obs = make_observability()
    rng = random.Random(seed)

    for difficulty in ["easy", "hard"]:
        scen = pick_scenario(spec, rng, difficulty)
        print(f"\n=== {difficulty.upper()} scenario  (THD={thd}, {n_runs} runs) ===")
        print(f"truth: {scen.truth}")
        print(f"{'variation':<9}{'final_acc':>16}{'actions':>16}")
        cfg = SimConfig(thd=thd)
        for v in VARIATIONS:
            r = run_variation_many(spec, obs, scen, cfg, v,
                                   n_runs=n_runs, seed=seed + 1)
            print(f"{v:<9}{r['final_mean']:>7.3f}+/-{r['final_sd']:<6.3f}"
                  f"{r['actions_mean']:>8.1f}+/-{r['actions_sd']:<5.1f}")


if __name__ == "__main__":
    main()

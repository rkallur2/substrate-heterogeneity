"""
sample_pool_domain17.py
=======================
Sample a pool of scenarios on domain17 (the controlled synthetic 17-node net),
rank by hardness, save top-K to disk. Mirrors sample_pool.py for the real net.

Notes on hardness for domain17: domain17 has no dedicated "Starts" target -- the
top-level rollup is SystemHealth (healthy/impaired/critical). We sample without
pinning and rank by count of non-nominal-state (non-state[0]) nodes, which is
the same proxy used for the real network.
"""
from __future__ import annotations
import random
import time
import pickle
import numpy as np
from collections import Counter

from domain17 import make_domain17
from simulation import sample_scenario


POOL_SIZE = 120
K_HARD = 8
POOL_SEED = 20260706
OUT_PATH = "/home/claude/substrate_abm/hard_scenarios_domain17.pkl"


def hardness(spec, s):
    return sum(1 for n, v in s.truth.items() if v != spec.nodes[n][0])


def main():
    spec = make_domain17()
    print(f"Sampling {POOL_SIZE} scenarios on domain17...")
    t0 = time.time()
    rng = random.Random(POOL_SEED)
    got = []
    for _ in range(POOL_SIZE):
        got.append(sample_scenario(spec, rng, name=f"pool{len(got)}"))
    dt = time.time() - t0
    print(f"Sampled {len(got)} in {dt:.1f}s\n")

    hardnesses = [hardness(spec, s) for s in got]
    print(f"Hardness distribution (out of {len(spec.nodes)}): "
          f"mean={np.mean(hardnesses):.2f} SD={np.std(hardnesses):.2f} "
          f"min={min(hardnesses)} max={max(hardnesses)}")
    ct = Counter(hardnesses)
    for h in sorted(ct.keys()):
        print(f"  h={h:2d}: {ct[h]:3d}  {'*' * min(ct[h], 40)}")

    ranked = sorted(enumerate(got), key=lambda kv: (-hardness(spec, kv[1]), kv[0]))
    hard = ranked[:K_HARD]

    print(f"\nTop {K_HARD} scenarios:")
    for rank, (idx, sc) in enumerate(hard):
        h = hardness(spec, sc)
        faults = [f"{n}={sc.truth[n]}" for n in spec.nodes if sc.truth[n] != spec.nodes[n][0]]
        print(f"  #{rank+1}: idx={idx}, h={h}")
        print(f"      faults: {', '.join(faults)}")

    # Also note the state of MainPower (the domain17 analogue of BatVolt) for
    # each hard scenario -- for the fault-state x aware_gap replication.
    payload = {
        "scenarios": [(idx, sc, hardness(spec, sc)) for idx, sc in hard],
        "pool_seed": POOL_SEED, "pool_size": len(got), "K_hard": K_HARD,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved to {OUT_PATH}")

    # print MainPower distribution across hard scenarios
    mp_counter = Counter(sc.truth.get("MainPower", "?") for _, sc, _ in payload["scenarios"])
    print(f"\nMainPower distribution in hard scenarios: {dict(mp_counter)}")


if __name__ == "__main__":
    main()

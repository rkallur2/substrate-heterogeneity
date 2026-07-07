"""
sample_pool.py
==============
Sample a pool of Starts=False scenarios on the real Kanno network, rank by
hardness, save top-K to disk. Split out so the (expensive) pool sampling only
runs once; the 2x2 experiment reads the saved pool.
"""
from __future__ import annotations
import random
import time
import pickle
import numpy as np

from domain_real import make_spec_shared
from simulation import sample_scenario


POOL_SIZE = 120
K_HARD = 8
POOL_SEED = 20260706
OUT_PATH = "/home/claude/substrate_abm/hard_scenarios.pkl"


def hardness(spec, s):
    return sum(1 for n, v in s.truth.items() if v != spec.nodes[n][0])


def main():
    spec = make_spec_shared("agent1")
    print(f"Sampling {POOL_SIZE} Starts=False scenarios on real Kanno net...")
    t0 = time.time()
    rng = random.Random(POOL_SEED)
    got = []
    tries = 0
    while len(got) < POOL_SIZE and tries < POOL_SIZE * 25:
        tries += 1
        sc = sample_scenario(spec, rng, name=f"pool{len(got)}")
        if sc.truth["Starts"] == "False":
            got.append(sc)
        if tries % 100 == 0:
            print(f"  ...{tries} tries, {len(got)} kept ({time.time()-t0:.1f}s)")
    dt = time.time() - t0
    print(f"Sampled {len(got)} in {tries} tries ({dt:.1f}s)\n")

    # rank
    hardnesses = [hardness(spec, s) for s in got]
    ranked = sorted(enumerate(got), key=lambda kv: (-hardness(spec, kv[1]), kv[0]))
    hard = ranked[:K_HARD]

    print(f"Top {K_HARD} scenarios by hardness:")
    for rank, (idx, sc) in enumerate(hard):
        h = hardness(spec, sc)
        faults = [f"{n}={sc.truth[n]}" for n in spec.nodes if sc.truth[n] != spec.nodes[n][0]]
        print(f"  #{rank+1}: pool-idx={idx}, h={h}")
        print(f"      faults: {', '.join(faults)}")

    payload = {
        "scenarios": [(idx, sc, hardness(spec, sc)) for idx, sc in hard],
        "pool_seed": POOL_SEED,
        "pool_size": len(got),
        "K_hard": K_HARD,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved to {OUT_PATH}")

    print(f"\nFull hardness distribution across pool of {len(got)}:")
    from collections import Counter
    ct = Counter(hardnesses)
    for h in sorted(ct.keys()):
        print(f"  h={h:2d}: {ct[h]:3d}  {'*' * ct[h]}")


if __name__ == "__main__":
    main()

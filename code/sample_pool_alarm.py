"""
sample_pool_alarm.py
====================
Sample a pool of scenarios on ALARM (Beinlich et al. 1989, 37 nodes), rank by
hardness, save top-K to disk. Mirrors sample_pool_domain17.py.

Hardness for ALARM: count of non-nominal-state nodes. Nominal for binary nodes
is FALSE (for diagnoses/errors) or NORMAL. Nominal for ordinal 3-state nodes is
NORMAL. For 4-state ventilator nodes, nominal is NORMAL. We use spec.nodes[n][0]
as the nominal-state proxy, matching the convention used for Kanno and Domain17.

Note: for ALARM this proxy will treat some nodes non-intuitively (e.g. FIO2's
states are {LOW, NORMAL} so LOW is nominal[0]; HYPOVOLEMIA states are
{TRUE, FALSE} so TRUE is nominal[0] — the reverse of what one might expect
clinically). This asymmetry is fine because the hardness measure is used only
to rank scenarios *relative* to each other for hard-scenario selection.
"""
from __future__ import annotations
import random
import time
import pickle
import os
import numpy as np
from collections import Counter

from alarm_domain import make_alarm, hardness_alarm
from simulation import sample_scenario


POOL_SIZE = 120
K_HARD = 8
POOL_SEED = 20260706
OUT_PATH = "./results/hard_scenarios_alarm.pkl"


def main():
    spec = make_alarm()
    print(f"Sampling {POOL_SIZE} scenarios on ALARM...")
    t0 = time.time()
    rng = random.Random(POOL_SEED)
    got = []
    for _ in range(POOL_SIZE):
        got.append(sample_scenario(spec, rng, name=f"pool{len(got)}"))
    dt = time.time() - t0
    print(f"Sampled {len(got)} in {dt:.1f}s")

    hardnesses = [hardness_alarm(spec, s) for s in got]
    print(f"\nHardness distribution (out of {len(spec.nodes)} nodes):")
    print(f"  mean={np.mean(hardnesses):.2f}  SD={np.std(hardnesses):.2f}  min={min(hardnesses)}  max={max(hardnesses)}")
    ct = Counter(hardnesses)
    for h in sorted(ct.keys()):
        print(f"  h={h:2d}: {ct[h]:3d}  {'*' * min(ct[h], 40)}")

    ranked = sorted(enumerate(got), key=lambda kv: (-hardness_alarm(spec, kv[1]), kv[0]))
    hard = ranked[:K_HARD]

    print(f"\nTop {K_HARD} hard scenarios selected:")
    scenarios_out = []
    for rank, (idx, sc) in enumerate(hard):
        h = hardness_alarm(spec, sc)
        # For clarity, show which diagnostic hypotheses are TRUE in each hard scenario
        active_dx = [d for d in ["HYPOVOLEMIA","LVFAILURE","ANAPHYLAXIS","INSUFFANESTH",
                                  "PULMEMBOLUS","KINKEDTUBE","DISCONNECT"]
                     if sc.truth[d] == "TRUE"]
        intubation = sc.truth.get("INTUBATION", "?")
        print(f"  #{rank+1}  idx={idx}  hardness={h}  active_dx={active_dx}  INTUBATION={intubation}")
        scenarios_out.append((idx, sc, h))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "scenarios": scenarios_out,
            "pool_seed": POOL_SEED,
            "pool_size": POOL_SIZE,
            "K_hard": K_HARD,
        }, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

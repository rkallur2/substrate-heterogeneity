"""
domain.py
=========
A synthetic causal-diagnosis domain (architecture sec. 7: build a structurally
equivalent system rather than depending on the proprietary Netica car network).

Phase-1 development domain: 8 nodes, mixed cardinalities, a small DAG with both
observable and inference-only nodes, and a 3-agent observability split with
overlaps (mirroring Okabe/Kanno Fig. 8).

The full study will scale this to ~17 nodes / 10 observable; the structure here
is the same, just smaller for fast iteration and validation.
"""

import numpy as np
from bbn_engine import CausalSpec


def make_domain() -> CausalSpec:
    nodes = {
        "PowerSource":   ["ok", "weak", "dead"],
        "Controller":    ["ok", "faulty"],
        "Signal":        ["strong", "weak", "none"],
        "SensorA":       ["ok", "faulty"],
        "SensorB":       ["ok", "faulty"],
        "Actuator":      ["ok", "degraded", "failed"],
        "Output":        ["nominal", "off"],
        "SystemHealth":  ["healthy", "impaired"],
    }
    parents = {
        "PowerSource":  [],
        "Controller":   [],
        "Signal":       ["PowerSource", "Controller"],
        "SensorA":      ["PowerSource"],
        "SensorB":      ["PowerSource"],
        "Actuator":     ["Signal", "Controller"],
        "Output":       ["Actuator", "Signal"],
        "SystemHealth": ["SensorA", "SensorB", "Actuator"],
    }

    cpts = {}
    cpts["PowerSource"] = np.array([0.75, 0.18, 0.07])
    cpts["Controller"] = np.array([0.88, 0.12])

    # Signal | PowerSource(ok,weak,dead) x Controller(ok,faulty) -> (strong,weak,none)
    cpts["Signal"] = np.array([
        [[0.90, 0.08, 0.02],   # power ok, ctrl ok
         [0.30, 0.50, 0.20]],  # power ok, ctrl faulty
        [[0.40, 0.45, 0.15],   # power weak, ctrl ok
         [0.10, 0.40, 0.50]],  # power weak, ctrl faulty
        [[0.02, 0.18, 0.80],   # power dead, ctrl ok
         [0.01, 0.09, 0.90]],  # power dead, ctrl faulty
    ])

    # SensorA | PowerSource -> (ok,faulty)
    cpts["SensorA"] = np.array([
        [0.95, 0.05],   # power ok
        [0.70, 0.30],   # power weak
        [0.20, 0.80],   # power dead
    ])
    # SensorB | PowerSource -> (ok,faulty)
    cpts["SensorB"] = np.array([
        [0.93, 0.07],
        [0.65, 0.35],
        [0.15, 0.85],
    ])

    # Actuator | Signal(strong,weak,none) x Controller(ok,faulty) -> (ok,degraded,failed)
    cpts["Actuator"] = np.array([
        [[0.90, 0.08, 0.02],   # signal strong, ctrl ok
         [0.55, 0.30, 0.15]],  # signal strong, ctrl faulty
        [[0.45, 0.40, 0.15],   # signal weak, ctrl ok
         [0.20, 0.45, 0.35]],  # signal weak, ctrl faulty
        [[0.05, 0.25, 0.70],   # signal none, ctrl ok
         [0.02, 0.13, 0.85]],  # signal none, ctrl faulty
    ])

    # Output | Actuator(ok,degraded,failed) x Signal(strong,weak,none) -> (nominal,off)
    cpts["Output"] = np.array([
        [[0.98, 0.02],   # act ok, sig strong
         [0.85, 0.15],   # act ok, sig weak
         [0.40, 0.60]],  # act ok, sig none
        [[0.75, 0.25],   # act degraded, sig strong
         [0.50, 0.50],
         [0.20, 0.80]],
        [[0.20, 0.80],   # act failed, sig strong
         [0.10, 0.90],
         [0.02, 0.98]],
    ])

    # SystemHealth | SensorA(ok,faulty) x SensorB(ok,faulty) x Actuator(ok,degraded,failed)
    #   -> (healthy, impaired)
    SH = np.zeros((2, 2, 3, 2))
    for a in range(2):      # SensorA
        for b in range(2):  # SensorB
            for act in range(3):  # Actuator
                # healthy if sensors ok and actuator ok-ish; degrade otherwise
                bad = (a == 1) + (b == 1) + (act >= 1)
                p_healthy = max(0.02, 0.95 - 0.30 * bad)
                SH[a, b, act] = [p_healthy, 1 - p_healthy]
    cpts["SystemHealth"] = SH

    return CausalSpec(nodes, parents, cpts)


def make_observability() -> dict:
    """3 agents, each observes a subset of the 'observable' nodes, with overlaps.
    Inference-only nodes (Signal, SystemHealth) are observable by no one and must
    be reasoned about -- the cooperative inference target."""
    return {
        "A": ["PowerSource", "SensorA", "Output"],
        "B": ["Controller", "SensorB", "Output"],
        "C": ["Actuator", "SensorA", "Controller"],
    }

"""
domain17.py
===========
Full-scale synthetic causal-diagnosis domain (architecture sec. 7): 17 nodes,
10 observable, with DELIBERATELY RICH MULTI-PARENT STRUCTURE so that the tree
engine's "drop secondary parents" approximation and the Markov engine's
single-parent skeleton both incur real, frequent loss -- unlike the 8-node dev
domain whose many single-parent links let the tree match the BBN exactly.

Design goals:
  * 17 nodes, mixed cardinalities (2-3 states).
  * >= half of non-root nodes have 2 OR 3 true parents (the bite).
  * 10 observable nodes split across 3 agents with overlaps (Okabe/Kanno Fig 8).
  * Unobservable inference targets that require cooperative reasoning.

Domain metaphor: a generic multi-subsystem device (power, signal, two sensor
chains, actuation, thermal, output, and rolled-up health) -- structurally like
the car-fault network but built fresh and fully under our control.
"""

import numpy as np
from bbn_engine import CausalSpec


def _cpt(shape, rng, sharpness=2.5):
    """Random but sensible CPT: Dirichlet-sampled conditionals with some
    sharpness so states are informative (not uniform)."""
    *parent_dims, k = shape
    arr = np.zeros(shape)
    if not parent_dims:
        a = rng.dirichlet(np.ones(k) * sharpness)
        return a
    it = np.ndindex(*parent_dims)
    for idx in it:
        # vary the Dirichlet concentration by parent index so parents matter
        bias = np.ones(k)
        for d, i in enumerate(idx):
            bias[(i + d) % k] += 1.5  # each parent nudges a different state
        arr[idx] = rng.dirichlet(bias * sharpness)
    return arr


def make_domain17(seed=20260624) -> CausalSpec:
    rng = np.random.default_rng(seed)

    nodes = {
        # roots
        "MainPower":     ["ok", "low", "dead"],          # 3
        "Clock":         ["ok", "drift"],                # 2
        "Firmware":      ["ok", "buggy"],                # 2
        # tier 1 (single/dual parent)
        "Regulator":     ["ok", "degraded", "failed"],   # parents: MainPower
        "BusVoltage":    ["nominal", "sag", "none"],     # parents: MainPower, Regulator
        "Timing":        ["good", "skewed"],             # parents: Clock, Firmware
        # tier 2 (multi-parent -- the bite)
        "SensorL":       ["ok", "noisy", "faulty"],      # parents: BusVoltage, Regulator
        "SensorR":       ["ok", "noisy", "faulty"],      # parents: BusVoltage, Timing
        "Comm":          ["up", "lossy", "down"],        # parents: BusVoltage, Firmware, Timing (3)
        "ControlLoop":   ["stable", "oscillating", "lost"], # parents: Timing, SensorL, SensorR (3)
        # tier 3 (multi-parent)
        "Actuator":      ["ok", "degraded", "failed"],   # parents: BusVoltage, ControlLoop
        "Thermal":       ["normal", "hot"],              # parents: Actuator, Regulator
        "DataPath":      ["clean", "corrupt"],           # parents: Comm, SensorL, SensorR (3)
        # tier 4 (deep multi-parent rollups)
        "Output":        ["nominal", "degraded", "off"], # parents: Actuator, DataPath, Comm (3)
        "Stability":     ["stable", "marginal", "unstable"], # parents: ControlLoop, Thermal
        "Integrity":     ["intact", "compromised"],      # parents: DataPath, Output
        "SystemHealth":  ["healthy", "impaired", "critical"], # parents: Output, Stability, Integrity (3)
    }

    parents = {
        "MainPower": [], "Clock": [], "Firmware": [],
        "Regulator":   ["MainPower"],
        "BusVoltage":  ["MainPower", "Regulator"],
        "Timing":      ["Clock", "Firmware"],
        "SensorL":     ["BusVoltage", "Regulator"],
        "SensorR":     ["BusVoltage", "Timing"],
        "Comm":        ["BusVoltage", "Firmware", "Timing"],
        "ControlLoop": ["Timing", "SensorL", "SensorR"],
        "Actuator":    ["BusVoltage", "ControlLoop"],
        "Thermal":     ["Actuator", "Regulator"],
        "DataPath":    ["Comm", "SensorL", "SensorR"],
        "Output":      ["Actuator", "DataPath", "Comm"],
        "Stability":   ["ControlLoop", "Thermal"],
        "Integrity":   ["DataPath", "Output"],
        "SystemHealth":["Output", "Stability", "Integrity"],
    }

    # root priors: skew toward "ok"-ish states so faults are the informative case
    cpts = {
        "MainPower": np.array([0.78, 0.16, 0.06]),
        "Clock":     np.array([0.85, 0.15]),
        "Firmware":  np.array([0.82, 0.18]),
    }
    # generated CPTs for the rest, shaped by parents
    for n, pa in parents.items():
        if n in cpts:
            continue
        shape = tuple(len(nodes[p]) for p in pa) + (len(nodes[n]),)
        cpts[n] = _cpt(shape, rng)

    return CausalSpec(nodes, parents, cpts)


def make_observability17() -> dict:
    """10 observable nodes split across 3 agents with overlaps. 7 nodes are
    UNobservable inference targets (must be reasoned about cooperatively):
       Timing, ControlLoop, DataPath, Stability, Integrity, Comm, SystemHealth.
    Observable (10): MainPower, Clock, Firmware, Regulator, BusVoltage,
       SensorL, SensorR, Actuator, Thermal, Output.
    """
    return {
        "A": ["MainPower", "Regulator", "SensorL", "Output"],
        "B": ["Clock", "BusVoltage", "SensorR", "Thermal"],
        "C": ["Firmware", "Actuator", "SensorL", "BusVoltage"],  # overlaps on SensorL, BusVoltage
    }


def make_observability17_n5() -> dict:
    """Team-size scalability partition: 5 agents, each with 2 observable nodes,
    two overlap pairs. 8 unique observable nodes; 9 unobservable inference targets
    (the original 7 unobservable + Regulator + Thermal, now hidden).

    Role assignments:
      A (Power supervisor):        MainPower, BusVoltage
      B (Timing / Firmware):       Clock, Firmware
      C (Left sensor operator):    SensorL, BusVoltage  -- overlap with A on BusVoltage
      D (Right sensor / actuation): SensorR, Actuator   -- overlap with E on Actuator
      E (Output supervisor):       Actuator, Output

    This is 2 observables per agent (vs 4 in N=3), so each agent has less
    direct information and must rely more on cooperative inference — the
    intended hardening for a team-size scalability test.
    """
    return {
        "A": ["MainPower", "BusVoltage"],
        "B": ["Clock", "Firmware"],
        "C": ["SensorL", "BusVoltage"],   # overlap with A on BusVoltage
        "D": ["SensorR", "Actuator"],     # overlap with E on Actuator
        "E": ["Actuator", "Output"],
    }


def parent_count_summary(spec):
    counts = {}
    for n, pa in spec.parents.items():
        counts[n] = len(pa)
    multi = [n for n, c in counts.items() if c >= 2]
    triple = [n for n, c in counts.items() if c >= 3]
    return counts, multi, triple

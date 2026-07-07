"""
domain_real.py
==============
The REAL Okabe/Kanno car-diagnosis domain, loaded from Kanno's actual Netica
files, wired for the substrate-heterogeneity study.

TWO EXPLICITLY SEPARATED CONFIGURATIONS (see KANNO_NETWORK_STATUS.md):

1. SHARED-FILE BASELINE (faithful "identical agents" replication):
   all three agents use ONE network file (agent1 by default -- best match to the
   paper's Figure-3 marginals). This is the anchor for every substrate comparison
   and depends on nothing Kanno has not already confirmed. Use make_spec_shared().

2. PARAMETRIC-HETEROGENEITY CONDITION (a SEPARATE, explicitly-bracketed
   investigation -- NOT the substrate manipulation):
   the three agents use agent0/agent1/agent2 respectively -- same 17-node
   structure, different CPTs. This studies whether CPT-level heterogeneity
   produces coordination costs analogous to substrate-level heterogeneity, as a
   candidate THIRD axis (parametric loss), kept methodologically distinct so a
   parametric difference never masquerades as a substrate effect.
   Use make_specs_parametric().

   IMPORTANT: whether the three files reflect the original simulation's intent is
   UNRESOLVED (email pending to Kanno). This condition is framed as OUR extension,
   NOT a claim about their setup. Do not attribute heterogeneous agents to the
   original study in any writeup until Kanno confirms.

OBSERVABILITY SPLIT: reconstructed from the paper's Figure 8. The 10 observable
nodes and their assignment to agents A/B/C (with overlaps) are mapped to the .dne
node identifiers below. MARKED PENDING CONFIRMATION -- verify against Figure 8 /
with Kanno. The 7 remaining nodes are unobservable inference targets.
"""

from netica_import import parse_dne
from bbn_engine import CausalSpec

# map paper's Figure-8 human names -> .dne node identifiers
# Figure 8 observable set (10): Alter, Main fuse, Lights, Starter Motor, Gas Tank,
#   Gas Filter, Spark plugs, Battery voltage, Air Filter, Fuel system
# NOTE: "Fuel system" (Fuel) is a derived node; Figure 8 places it in the shared
#   region. Battery voltage (BatVolt) and Main fuse (MFuse) likewise. This mapping
#   is our best reconstruction of Fig 8 and is PENDING CONFIRMATION.
OBSERVABLE_MAP = {
    "A": ["StMotor", "GasTank", "Alter", "Lights"],       # A's region + shared
    "B": ["Plugs", "BatVolt", "MFuse", "Lights"],         # B's region + shared
    "C": ["AirFilter", "Fuel", "GasFilter", "MFuse"],     # C's region + shared
}
# -> observable union = {StMotor, GasTank, Alter, Lights, Plugs, BatVolt, MFuse,
#    AirFilter, Fuel, GasFilter} = 10 nodes.  Unobservable (7): PlugVolt, Starter,
#    Dist, SpkQual, Air, Timing, Starts.


def make_spec_shared(which: str = "agent1") -> CausalSpec:
    """Faithful baseline: one shared network for all agents."""
    return parse_dne(f"{which}.dne")


def make_specs_parametric() -> dict:
    """Parametric-heterogeneity condition: agent-specific networks.
    Returns {agent_name: CausalSpec}. SEPARATE from substrate manipulation."""
    return {
        "A": parse_dne("agent0.dne"),
        "B": parse_dne("agent1.dne"),
        "C": parse_dne("agent2.dne"),
    }


def make_observability_real() -> dict:
    """Per-agent observable nodes (reconstructed from Figure 8, PENDING CONFIRM)."""
    return {k: list(v) for k, v in OBSERVABLE_MAP.items()}

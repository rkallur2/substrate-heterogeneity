"""
alarm_domain.py
===============
ALARM network (Beinlich et al. 1989) loaded from the canonical bnlearn BIF file.

37 nodes, 46 edges, max in-degree 4.

Beinlich's canonical categorization: 8 diagnostic hypotheses, 16 findings,
13 intermediate physiological states.

In this module we adopt a slightly refined observability: 14 findings observed
by three agents in the anesthesia team, with ERRLOWOUTPUT and ERRCAUTER treated
as intermediate rather than observable (they are monitor-fidelity indicators,
not direct clinical observations). The resulting 8 + 14 + 15 = 37 partition.

ALARM's variable-state orderings in the BIF file are NOT consistent with respect
to "nominal" — e.g. HYPOVOLEMIA has states {TRUE, FALSE} with TRUE as state[0],
while HISTORY has states {TRUE, FALSE} where state[0]=TRUE is clinically the
positive finding. For Kanno and Domain17, state[0] is always the nominal state
by construction, so hardness = count of non-nominal states works out of the box.
For ALARM we need an explicit clinical-nominal map, provided as CLINICAL_NOMINAL
below, and hardness_alarm() uses it.
"""
import os
import re
import numpy as np
from bbn_engine import CausalSpec

_BIF_PATH = os.path.join(os.path.dirname(__file__), "alarm.bif")


def _parse_bif(bif_text):
    """Parse BIF format into (variables, cpts) matching CausalSpec conventions.
       cpts[node] has shape (parent1_card, ..., self_card) or (self_card,) for roots."""
    variables = {}
    var_pat = re.compile(
        r"variable\s+(\w+)\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{\s*([^}]+)\}",
        re.MULTILINE)
    for m in var_pat.finditer(bif_text):
        name, k, states_str = m.group(1), int(m.group(2)), m.group(3)
        states = [s.strip() for s in states_str.split(",")]
        assert len(states) == k
        variables[name] = states

    cpts = {}
    parents_map = {}
    prob_pat = re.compile(
        r"probability\s*\(\s*(\w+)\s*(?:\|\s*([\w,\s]+?))?\s*\)\s*\{(.+?)\}",
        re.DOTALL)
    for m in prob_pat.finditer(bif_text):
        node = m.group(1)
        parents_str = m.group(2)
        body = m.group(3)
        parents = [p.strip() for p in parents_str.split(",")] if parents_str else []
        parents_map[node] = parents

        node_card = len(variables[node])
        parent_cards = [len(variables[p]) for p in parents]

        if not parents:
            tbl_m = re.search(r"table\s+([\d.,\s]+);", body)
            probs = [float(x) for x in tbl_m.group(1).split(",")]
            cpts[node] = np.array(probs)
        else:
            row_pat = re.compile(r"\(\s*([\w,\s]+?)\s*\)\s+([\d.,\s]+?);")
            table = np.zeros(parent_cards + [node_card])
            for rm in row_pat.finditer(body):
                p_state_names = [s.strip() for s in rm.group(1).split(",")]
                probs = [float(x) for x in rm.group(2).split(",")]
                p_idx = tuple(variables[parents[i]].index(p_state_names[i])
                              for i in range(len(parents)))
                table[p_idx] = probs
            cpts[node] = table

    return variables, cpts, parents_map


def make_alarm(bif_path=None) -> CausalSpec:
    """Load the ALARM network from its BIF file into a CausalSpec.
    All 37 nodes; CPT values are Beinlich et al. 1989 originals (no resampling).
    """
    if bif_path is None:
        bif_path = _BIF_PATH
    with open(bif_path) as f:
        bif_text = f.read()
    variables, cpts, parents = _parse_bif(bif_text)

    # CausalSpec expects 'nodes' as {name: [state_names]}, 'parents' as {name: [parent_names]},
    # 'cpts' as {name: ndarray of shape (parent1_card, ..., self_card)}.
    return CausalSpec(variables, parents, cpts)


def make_observability_alarm() -> dict:
    """14 observable nodes split across 3 agents with 2 nodes of overlap,
    reflecting real anesthesia team roles.

    Agent A (Anesthesiologist): hemodynamic monitors — 6 nodes.
    Agent B (Respiratory technician): ventilator and gas exchange — 5 nodes.
    Agent C (Circulating nurse / monitor operator): EKG, saturation, history — 5 nodes.

    Overlaps: SAO2 (A + C), MINVOLSET (B + C).
    Total unique observable nodes: 14.
    Total hidden inference targets: 23 (8 diagnoses + 15 intermediate).

    ERRLOWOUTPUT and ERRCAUTER are treated as intermediate rather than
    observable — they are monitor-fidelity indicators inferred from
    monitor discrepancies rather than direct clinical observations.
    """
    return {
        "A": ["BP", "CVP", "PCWP", "HRBP", "PAP", "SAO2"],
        "B": ["EXPCO2", "MINVOL", "MINVOLSET", "PRESS", "FIO2"],
        "C": ["HREKG", "HRSAT", "HISTORY", "SAO2", "MINVOLSET"],
    }


def make_observability_alarm_n5() -> dict:
    """Team-size scalability partition for ALARM: 5 agents, 2 obs/agent,
    2 overlaps. 8 unique observed nodes; 6 previously-observable nodes
    (PCWP, HRBP, HRSAT, MINVOLSET, FIO2, HISTORY) move to hidden inference
    targets. Total hidden at N=5: 29 (6 newly + 23 pre-existing).

    Roles (real anesthesia-team decomposition, extended to 5 members):
      A (Anesthesiologist):          BP, PAP
      B (Cardio Assistant):          CVP, HREKG
      C (Respiratory Tech):          EXPCO2, MINVOL
      D (Airway Pressure specialist): PRESS, MINVOL   -- overlap with C
      E (Oximeter Operator):         SAO2, HREKG      -- overlap with B
    """
    return {
        "A": ["BP", "PAP"],
        "B": ["CVP", "HREKG"],
        "C": ["EXPCO2", "MINVOL"],
        "D": ["PRESS", "MINVOL"],   # overlap with C on MINVOL
        "E": ["SAO2", "HREKG"],     # overlap with B on HREKG
    }


# Category labels for reference (used in the manuscript's §3.7.5 write-up):
DIAGNOSES = [
    "HYPOVOLEMIA", "LVFAILURE", "ANAPHYLAXIS", "INSUFFANESTH",
    "PULMEMBOLUS", "INTUBATION", "KINKEDTUBE", "DISCONNECT",
]
OBSERVABLES = [
    "BP", "CVP", "PCWP", "HRBP", "PAP", "SAO2",
    "EXPCO2", "MINVOL", "MINVOLSET", "PRESS", "FIO2",
    "HREKG", "HRSAT", "HISTORY",
]
INTERMEDIATE = [
    "ARTCO2", "CATECHOL", "CO", "ERRCAUTER", "ERRLOWOUTPUT",
    "HR", "LVEDVOLUME", "PVSAT", "SHUNT", "STROKEVOLUME",
    "TPR", "VENTALV", "VENTLUNG", "VENTMACH", "VENTTUBE",
]


# Clinical nominal-state map for hardness ranking.
# For a well-monitored anesthesia patient with no active faults, nominal means:
# - No disease/fault diagnoses (all diagnostic hypotheses FALSE)
# - Normal intubation (not esophageal or one-sided)
# - No monitor errors (ERRLOWOUTPUT/ERRCAUTER FALSE)
# - No clinical history of ventricular failure (HISTORY FALSE)
# - All ordinal physiological measurements NORMAL
# - Normal shunt fraction (NORMAL, not HIGH)
# - Normal catecholamine level (NORMAL, not HIGH stress response)
# - Normal FiO2 (breathing normal air)
CLINICAL_NOMINAL = {
    # Diagnostic hypotheses (fault = TRUE, nominal = FALSE)
    "HYPOVOLEMIA": "FALSE", "LVFAILURE": "FALSE", "ANAPHYLAXIS": "FALSE",
    "INSUFFANESTH": "FALSE", "PULMEMBOLUS": "FALSE",
    "KINKEDTUBE": "FALSE", "DISCONNECT": "FALSE",
    # Discrete but multi-state
    "INTUBATION": "NORMAL",
    # Errors (nominal = no error = FALSE)
    "ERRLOWOUTPUT": "FALSE", "ERRCAUTER": "FALSE",
    # Clinical history (nominal = no positive history = FALSE)
    "HISTORY": "FALSE",
    # Binary intermediate states
    "SHUNT": "NORMAL", "CATECHOL": "NORMAL",
    # Ordinal 3-state physiological measurements (nominal = NORMAL)
    "CVP": "NORMAL", "PCWP": "NORMAL", "LVEDVOLUME": "NORMAL",
    "STROKEVOLUME": "NORMAL", "HRBP": "NORMAL", "HREKG": "NORMAL",
    "HRSAT": "NORMAL", "TPR": "NORMAL", "PVSAT": "NORMAL",
    "SAO2": "NORMAL", "PAP": "NORMAL", "ARTCO2": "NORMAL",
    "HR": "NORMAL", "CO": "NORMAL", "BP": "NORMAL",
    # 4-state ventilator/gas nodes (nominal = NORMAL)
    "EXPCO2": "NORMAL", "MINVOL": "NORMAL", "PRESS": "NORMAL",
    "VENTMACH": "NORMAL", "VENTTUBE": "NORMAL",
    "VENTLUNG": "NORMAL", "VENTALV": "NORMAL",
    # Ventilator setting (nominal = NORMAL, the standard clinical setting)
    "MINVOLSET": "NORMAL",
    # FIO2 (nominal = NORMAL air; LOW would indicate supplemental O2 setting adjusted)
    "FIO2": "NORMAL",
}


def hardness_alarm(spec, scenario) -> int:
    """Count of nodes in non-nominal (clinically abnormal or fault) states."""
    return sum(1 for n, v in scenario.truth.items() if v != CLINICAL_NOMINAL[n])

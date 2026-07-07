"""
netica_import.py
================
Convert Kanno's actual Netica .dne car-diagnosis network (the real network from
Okabe/Kanno CTW 2025) into a CausalSpec, so the study can run on the authoritative
domain rather than the synthetic domain17.

Netica CPT layout: probs are listed with the LAST parent varying fastest, then
the node's own states innermost. We reshape into (parent0, parent1, ..., node)
to match CausalSpec's convention (last axis = node), which is exactly Netica's
ordering, so the raw flat array reshapes directly.
"""
import re
import numpy as np
from bbn_engine import CausalSpec, BBNEngine


def parse_dne(path):
    txt = open(path).read()
    node_blocks = re.findall(r'node (\w+) \{(.*?)\n\t\};', txt, re.DOTALL)

    nodes, parents, raw_probs = {}, {}, {}
    for name, body in node_blocks:
        states = re.search(r'states = \((.*?)\);', body).group(1)
        states = [s.strip() for s in states.split(',')]
        nodes[name] = states
        pa = re.search(r'parents = \((.*?)\);', body).group(1).strip()
        parents[name] = [p.strip() for p in pa.split(',') if p.strip()] if pa else []
        # extract the probability numbers (strip comments first).
        # The probs assignment ends at the ';' that closes it; the following
        # field varies (numcases / title / whenchanged / belief), so match up to
        # the next field keyword or the closing brace rather than assuming numcases.
        pmatch = re.search(r'probs =\s*(.*?);\s*(?:numcases|title|whenchanged|belief|functable|equation|\})',
                           body, re.DOTALL)
        pblock = pmatch.group(1)
        pblock = re.sub(r'//.*', '', pblock)          # remove // comments
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', pblock)
        raw_probs[name] = [float(x) for x in numbers]

    # build CPT ndarrays with shape (|pa0|,...,|paN|, |node|)
    cpts = {}
    for n in nodes:
        pa = parents[n]
        k = len(nodes[n])
        shape = tuple(len(nodes[p]) for p in pa) + (k,)
        arr = np.array(raw_probs[n], dtype=float).reshape(shape)
        # normalize each conditional row (Netica values are near-normalized already)
        arr = arr / arr.sum(axis=-1, keepdims=True)
        cpts[n] = arr

    return CausalSpec(nodes, parents, cpts)


if __name__ == "__main__":
    spec = parse_dne("agent1.dne")
    print("Parsed CausalSpec: %d nodes" % len(spec.nodes))
    # validate: construct BBN, check priors are sane and inference runs
    e = BBNEngine(spec)
    print("\nMarginal priors (a few):")
    for n in ["MFuse", "BatVolt", "Starts", "SpkQual"]:
        print(" ", n, {k: round(v, 3) for k, v in e.posterior(n).items()})
    # multi-parent inference sanity: force a no-start condition, see causes shift
    print("\nInference check: observe Starts=False, Lights=dim")
    e2 = BBNEngine(spec)
    e2.ingest("Starts", "False"); e2.ingest("Lights", "dim")
    for n in ["BatVolt", "Alter", "Fuel", "SpkQual"]:
        print(" ", n, {k: round(v, 3) for k, v in e2.posterior(n).items()})

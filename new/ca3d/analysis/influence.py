"""Coupling tests for the hierarchical CA: is a layer's influence real or decorative?

Most functions take `make`, a zero-argument function returning a fresh
HierarchicalCA (e.g. `cityscape.make`), so every variant starts from
identical rules and states and only the intervention differs.

  pin_layers          Freeze each layer at its initial state and measure how much of the
                      fine space-time changes.  The bluntest and most informative test;
                      it also changes the layer's statistics, so a nonzero result
                      confirms influence without quantifying it.
  perturb_layers      Flip one cell of each layer at t = 0 and measure how far the
                      difference spreads into the fine layer.  0.000 means the channel
                      is severed; values around 0.01-0.1 mean it is live.
  pattern_traffic     How often each (context, neighbourhood pattern) is actually used.
  parent_sensitivity  Per parent bit, the fraction of rule entries whose output depends
                      on it: uniformly over entries, or weighted by traffic.  Dynamics
                      often visit only a few patterns, so the two can disagree wildly:
                      a rule table can differ between contexts in 60% of its entries
                      yet almost never at the entries that are actually used.
  xor_coupled         Rule banks built so each parent bit inverts the output on chosen
                      patterns: effective sensitivity 1.0 whatever the traffic.

Ablation (removing a layer outright) needs no helper: build both hierarchies and
compare their measures with search.sweep.  Removing a layer usually changes more than
one thing (it also halves the context count), so state the confound.
"""
import dataclasses

import numpy as np

from ..rulesets import life


def _difference(a, b):
    return float((a != b).mean())


def pin_layers(make, steps, layers=None):
    """Rows {layer, differs, density}: the fraction of fine space-time voxels that
    change when `layer` never updates (is "pinned" at its initial state), and the fine
    density of that run.  The first row, layer 'none', is the unpinned baseline."""
    base = make().run(steps).fine
    rows = [{'layer': 'none', 'differs': 0.0, 'density': float(base.mean())}]
    ca = make()
    for i in range(1, ca.n_layers) if layers is None else layers:
        ca = make()
        ca.layers[i] = dataclasses.replace(ca.layers[i], pinned=True)
        fine = ca.run(steps).fine
        rows.append({'layer': i, 'differs': _difference(fine, base), 'density': float(fine.mean())})
    return rows


def perturb_layers(make, steps, layers=None, cell=None):
    """Rows {layer, differs}: flip one cell of `layer` (default: its middle cell) at
    t = 0 and measure the fraction of fine space-time voxels that end up different."""
    base = make().run(steps).fine
    rows = []
    ca = make()
    for i in range(ca.n_layers) if layers is None else layers:
        ca = make()
        where = tuple(s // 2 for s in ca.states[i].shape) if cell is None else cell
        ca.states[i][where] ^= 1
        rows.append({'layer': i, 'differs': _difference(ca.run(steps).fine, base)})
    return rows


def pattern_traffic(ca, steps, layer=0):
    """Run `ca` for `steps` steps (in place) and count, on every step where `layer`
    updates, how many of its cells read each (context, pattern).  Shape (n_ctx, P)."""
    P = ca.nb.n_patterns
    n_ctx = 2 ** len(ca.parents[layer])
    counts = np.zeros(n_ctx * P, np.int64)
    for _ in range(steps):
        if ca.layers[layer].fires(ca.t):
            code = ca.context(layer) * P + ca.nb.encode(ca.states[layer])
            counts += np.bincount(code.ravel(), minlength=n_ctx * P)
        ca.step()
    return counts.reshape(n_ctx, P)


def parent_sensitivity(rules, traffic=None):
    """Per parent (nearest first, i.e. the context's most significant bit first): the
    fraction of entries rules[c, p] that change when that parent's bit flips.

    Without traffic every (context, pattern) counts equally (*uniform* sensitivity);
    with traffic each counts by how often the dynamics visit it (*effective*)."""
    rules = np.asarray(rules)
    n_ctx = rules.shape[0]
    n_parents = n_ctx.bit_length() - 1
    out = []
    for j in range(n_parents):
        partner = np.arange(n_ctx) ^ (1 << (n_parents - 1 - j))
        differs = rules != rules[partner]
        if traffic is None:
            out.append(float(differs.mean()))
        else:
            out.append(float((traffic * differs).sum() / max(traffic.sum(), 1)))
    return np.array(out)


def xor_coupled(base, n_parents=1, patterns=None, rng=None, strength=None):
    """Banks (2**n_parents, P) where each parent bit inverts `base` on chosen patterns:
    rules[c, p] = base[p] XOR (popcount(c) odd and p chosen).  By default every pattern
    is chosen (strength = P), which forces effective sensitivity 1.0, bluntly: the
    conditioned output becomes the exact inverse rather than a different regime."""
    base = np.asarray(base, np.uint8)
    P = base.size
    if patterns is None:
        patterns = np.arange(P) if strength is None or strength >= P else \
            rng.choice(P, strength, replace=False)
    chosen = np.zeros(P, np.uint8)
    chosen[patterns] = 1
    odd = (life.POPCOUNT[np.arange(2 ** n_parents)] % 2).astype(np.uint8)
    return base[None, :] ^ (odd[:, None] & chosen[None, :])
